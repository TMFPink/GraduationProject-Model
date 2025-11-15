import os, io, tempfile
import time
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from ultralytics import YOLO
from pinecone import Pinecone
import open_clip
from dotenv import load_dotenv
load_dotenv()


# --------------------
# Environment / config
# --------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/card_model_29_10.pt")
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "ViT-H-14")
CLIP_PRETRAINED = os.getenv("CLIP_PRETRAINED", "laion2B-s32B-b79K")
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ygo-embedded")
CLIP_METADATA_VALUE = os.getenv(
    "CLIP_METADATA_VALUE",
    "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------
# YOLO & CLIP singletons
# --------------------
_yolo = None
_clip = None
_clip_pre = None
_pc = None
_index = None

def load_singletons():
    global _yolo, _clip, _clip_pre, _pc, _index
    
    print("Loading singletons...")
    start_time = time.time()

    if _clip is None:
        print("Loading CLIP model...")
        clip_start = time.time()
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
        )
        _clip = clip_model.to(device).eval()
        _clip_pre = clip_preprocess
        print(f"CLIP model loaded in {time.time() - clip_start:.2f}s")

    if _yolo is None:
        print("Loading YOLO model...")
        yolo_start = time.time()
        _yolo = YOLO(MODEL_PATH)
        print(f"YOLO model loaded in {time.time() - yolo_start:.2f}s")

    if _pc is None:
        print("Connecting to Pinecone...")
        pc_start = time.time()
        _pc = Pinecone(api_key=PINECONE_API_KEY)
        _index = _pc.Index(PINECONE_INDEX_NAME)
        print(f"Pinecone connected in {time.time() - pc_start:.2f}s")

    total_time = time.time() - start_time
    print(f"All singletons loaded in {total_time:.2f}s")
    return _yolo, _clip, _clip_pre, _index

# --------------------
# Constants & helpers
# --------------------
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

LABEL_FILTER_RULES = {
    "normal-monster": [
        "Normal Monster","Normal Tuner Monster","Effect Monster","Flip Effect Monster",
        "Flip Tuner Effect Monster","Gemini Monster","Pendulum Effect Monster",
        "Pendulum Effect Fusion Monster","Pendulum Effect Ritual Monster",
        "Pendulum Flip Effect Monster","Pendulum Tuner Effect Monster",
        "Spirit Monster","Tuner Monster","Union Effect Monster",
    ],
    "effect-monster": [],
    "ritual-monster": ["Ritual Monster","Ritual Effect Monster","Pendulum Effect Ritual Monster"],
    "fusion-monster": ["Fusion Monster","Pendulum Effect Fusion Monster"],
    "synchro-monster": ["Synchro Monster","Synchro Pendulum Effect Monster"],
    "xyz-monster": ["XYZ Monster","XYZ Pendulum Effect Monster"],
    "link-monster": ["Link Monster"],
    "spell-card": ["Spell Card"],
    "trap-card": ["Trap Card"],
}

def _clip_target_hw(model) -> Tuple[int, int]:
    sz = getattr(model.visual, "image_size", 224)
    if isinstance(sz, (tuple, list)):
        return int(sz[0]), int(sz[1])
    return int(sz), int(sz)

def _letterbox_square_rgb(rgb: np.ndarray, size: int, pad_val=114):
    h, w = rgb.shape[:2]
    s = max(h, w)
    canvas = np.full((s, s, 3), pad_val, dtype=rgb.dtype)
    y0 = (s - h) // 2
    x0 = (s - w) // 2
    canvas[y0:y0+h, x0:x0+w] = rgb
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)

@torch.no_grad()
def embed_image_letterbox(img_pil: Image.Image, clip_model) -> np.ndarray:
    print("Embedding image with CLIP...")
    embed_start = time.time()
    
    Ht, Wt = _clip_target_hw(clip_model)
    side = max(Ht, Wt)
    rgb = np.asarray(img_pil.convert("RGB"))
    sq = _letterbox_square_rgb(rgb, size=side, pad_val=114)
    if Ht != Wt:
        sq = cv2.resize(sq, (int(Wt), int(Ht)), interpolation=cv2.INTER_AREA)

    tensor = torch.from_numpy(sq).permute(2,0,1).float() / 255.0
    for c,(m,s) in enumerate(zip(CLIP_MEAN, CLIP_STD)):
        tensor[c] = (tensor[c] - m) / s
    tensor = tensor.unsqueeze(0).to(device)

    feat = clip_model.encode_image(tensor)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    
    embed_time = time.time() - embed_start
    print(f"===============> Image embedded in {embed_time:.3f}s")
    return feat.float().cpu().numpy().flatten().astype(np.float32)

def preprocess_query(img_pil: Image.Image) -> Image.Image:
    print("Preprocessing query image...")
    img_pil = img_pil.convert("RGB")
    img_pil = ImageOps.exif_transpose(img_pil)
    w, h = img_pil.size
    crop_box = (int(0.05*w), int(0.05*h), int(0.95*w), int(0.95*h))
    result = img_pil.crop(crop_box)
    print(f"Image preprocessed: {w}x{h} -> {result.size[0]}x{result.size[1]}")
    return result

def detect_cards_yolo(yolo, img_bgr, conf=0.6):
    print(f"Running YOLO detection with confidence {conf}...")
    detect_start = time.time()
    
    res = yolo(img_bgr, conf=conf, verbose=False)
    boxes = res[0].boxes
    names = yolo.names if hasattr(yolo, "names") else None
    
    detect_time = time.time() - detect_start
    num_detections = len(boxes) if boxes is not None else 0
    print(f"===============> YOLO detected {num_detections} cards in {detect_time:.3f}s")
    return boxes, names

def crop_with_padding(img, box, pad=20):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img.shape[1], x2 + pad)
    y2 = min(img.shape[0], y2 + pad)
    return img[y1:y2, x1:x2]

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_card(image, debug_view=False, min_area_abs=5000, min_box_ratio=40.0):
    print("Attempting perspective correction...")
    warp_start = time.time()
    
    def poly_area(pts):
        x = pts[:,0]; y = pts[:,1]
        return 0.5 * abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 50, 50)
    edges = cv2.Canny(gray, 60, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), 1)

    H, W = gray.shape
    img_area = H * W
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        if cv2.contourArea(c) < min_area_abs:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        quad = approx.reshape(-1,2).astype(np.float32) if len(approx)==4 else cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
        quad = order_points(quad)

        x, y, w_box, h_box = cv2.boundingRect(quad.astype(np.int32))
        bbox_ratio = (w_box * h_box) / (img_area + 1e-6) * 100.0
        if bbox_ratio < min_box_ratio:
            continue

        # validity checks
        if np.any(quad[:,0] < -2) or np.any(quad[:,1] < -2) or np.any(quad[:,0] > W+2) or np.any(quad[:,1] > H+2):
            continue
        if len(np.unique(quad, axis=0)) < 4:
            continue
        if cv2.isContourConvex(quad.astype(np.int32)) is False:
            continue
        A = poly_area(quad)
        if A < 0.02 * (W*H):
            continue

        bb = cv2.boundingRect(quad.astype(np.int32))
        rect_area = bb[2]*bb[3]
        rect_score = A / (rect_area + 1e-6)
        candidates.append((rect_score*A, quad))

    if not candidates:
        print("No valid card contours found, using original crop")
        return image, gray, edges

    _, quad = max(candidates, key=lambda t: t[0])
    (tl, tr, br, bl) = quad
    width = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    height = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    width = max(width, 10); height = max(height, 10)

    dst = np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, M, (width, height),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)

    if warped.size == 0 or np.std(warped) < 2.0:
        print("Warped image is invalid, using original")
        return image, gray, edges
    
    warp_time = time.time() - warp_start
    print(f"===============> Perspective correction completed in {warp_time:.3f}s")
    return warped, gray, edges

# --------------------
# Pinecone search
# --------------------
def search_by_image_with_filter(img_pil: Image.Image, label: Optional[str], top_k: int, clip_model, index):
    # print(f"Searching Pinecone for similar cards (label: {label}, top_k: {top_k})...")
    search_start = time.time()
    
    img_pil = preprocess_query(img_pil)
    qvec = embed_image_letterbox(img_pil, clip_model)

    pinecone_filter = None
    if label and label in LABEL_FILTER_RULES:
        pinecone_filter = {
            "card_type": {"$in": LABEL_FILTER_RULES[label]},
            "model": {"$eq": CLIP_METADATA_VALUE}
        }
        # print(f"Applied filter for card type: {label}")

    results = index.query(
        vector=qvec.tolist(),
        namespace="image",
        top_k=1,
        include_metadata=True,
        filter=pinecone_filter,
    )
    matches = results.get("matches") or []
    print(f"Pinecone returned {len(matches)} matches")
    
    if not matches:
        print("No matches found in Pinecone")
        return []

    grouped = {}
    for m in matches:
        meta = m["metadata"]
        cid = meta["card_id"]
        score = m["score"]
        if cid not in grouped or score > grouped[cid]["score"]:
            grouped[cid] = {**meta, "score": score}

    final_results = sorted(grouped.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    search_time = time.time() - search_start
    print(f"===============> Search completed in {search_time:.3f}s, returning {len(final_results)} unique cards")
    
    if final_results:
        best_match = final_results[0]
        print(f"Best match: {best_match.get('name', 'Unknown')} (score: {best_match.get('score', 0):.4f})")
    
    return final_results

# --------------------
# Public API (used by FastAPI)
# --------------------
def detect_card_names(image_path: str, conf: float = 0.6, top_k: int = 5) -> List[str]:
    print(f"\nStarting card name detection for: {os.path.basename(image_path)}")
    total_start = time.time()
    
    yolo, clip_model, _, index = load_singletons()

    print("Loading image...")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Cannot read image: {image_path}")
        raise ValueError(f"Cannot read image: {image_path}")
    
    print(f"Image loaded: {img_bgr.shape[1]}x{img_bgr.shape[0]}")

    max_size = 1280
    if max(img_bgr.shape[:2]) > max_size:
        # print(f"Resizing image from {img_bgr.shape[1]}x{img_bgr.shape[0]} to fit {max_size}px")
        scale = max_size / max(img_bgr.shape[:2])
        img_bgr = cv2.resize(img_bgr, (int(img_bgr.shape[1]*scale), int(img_bgr.shape[0]*scale)))
        # print(f"Image resized to: {img_bgr.shape[1]}x{img_bgr.shape[0]}")

    boxes, class_names = detect_cards_yolo(yolo, img_bgr, conf)
    if len(boxes) == 0:
        print("No cards detected")
        return []

    names: List[str] = []
    for i, box in enumerate(boxes):
        print(f"\nProcessing card {i+1}/{len(boxes)}...")
        
        cls_id = int(box.cls[0]) if hasattr(box, "cls") else -1
        label = class_names[cls_id] if (class_names and 0 <= cls_id < len(class_names)) else f"Card {i+1}"
        print(f"Card type: {label}")

        print("Cropping card from detection...")
        crop_bgr = crop_with_padding(img_bgr, box)
        print(f"Cropped to: {crop_bgr.shape[1]}x{crop_bgr.shape[0]}")

        # warp if aspect looks wrong
        h, w = crop_bgr.shape[:2]
        expect = 1.46; tol = 0.1
        r_hw, r_wh = h / w, w / h
        if (expect - tol) <= r_hw <= (expect + tol) or (expect - tol) <= r_wh <= (expect + tol):
            # print("Aspect ratio is good, skipping perspective correction")
            warped_bgr = crop_bgr
        else:
            # print(f"Aspect ratio {r_hw:.2f} needs correction (expected ~{expect})")
            warped_bgr, _, _ = warp_card(crop_bgr, debug_view=False)

        if warped_bgr.shape[1] > warped_bgr.shape[0]:
            # print("Rotating card to portrait orientation")
            warped_bgr = cv2.rotate(warped_bgr, cv2.ROTATE_90_CLOCKWISE)

        # try 0 and 180
        # print("Testing card orientations (0° and 180°)...")
        candidates = [warped_bgr, cv2.rotate(warped_bgr, cv2.ROTATE_180)]
        best = None; best_score = -1.0
        
        for j, cand in enumerate(candidates):
            angle = "0°" if j == 0 else "180°"
            # print(f"Testing orientation {angle}...")
            pil = Image.fromarray(cv2.cvtColor(cand, cv2.COLOR_BGR2RGB))
            res = search_by_image_with_filter(pil, label=label, top_k=top_k, clip_model=clip_model, index=index)
            if res and float(res[0].get("score", 0)) > best_score:
                best, best_score = res[0], float(res[0].get("score", 0))
                # print(f"New best orientation: {angle} (score: {best_score:.4f})")
        
        if best:
            card_name = best.get("name")
            names.append(card_name)
            print(f"Card {i+1} identified: {card_name}")
        else:
            print(f"Card {i+1} could not be identified")
    
    total_time = time.time() - total_start
    print(f"\n===============> Detection completed in {total_time:.2f}s")
    print(f"Final results: {names}")
    return names

def detect_card_ids(image_path: str, conf: float = 0.6, top_k: int = 5) -> List[str]:
    print(f"\nStarting card ID detection for: {os.path.basename(image_path)}")
    total_start = time.time()
    
    yolo, clip_model, _, index = load_singletons()

    print("Loading image...")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Cannot read image: {image_path}")
        raise ValueError(f"Cannot read image: {image_path}")
    
    print(f"Image loaded: {img_bgr.shape[1]}x{img_bgr.shape[0]}")

    max_size = 1280
    if max(img_bgr.shape[:2]) > max_size:
        # print(f"Resizing image from {img_bgr.shape[1]}x{img_bgr.shape[0]} to fit {max_size}px")
        scale = max_size / max(img_bgr.shape[:2])
        img_bgr = cv2.resize(img_bgr, (int(img_bgr.shape[1]*scale), int(img_bgr.shape[0]*scale)))
        # print(f"Image resized to: {img_bgr.shape[1]}x{img_bgr.shape[0]}")

    boxes, class_names = detect_cards_yolo(yolo, img_bgr, conf)
    if len(boxes) == 0:
        print("No cards detected")
        return []

    ids: List[str] = []
    for i, box in enumerate(boxes):
        print(f"\nProcessing card {i+1}/{len(boxes)}...")
        
        cls_id = int(box.cls[0]) if hasattr(box, "cls") else -1
        label = class_names[cls_id] if (class_names and 0 <= cls_id < len(class_names)) else f"Card {i+1}"
        print(f"Card type: {label}")

        # print("Cropping card from detection...")
        crop_bgr = crop_with_padding(img_bgr, box)
        # print(f"Cropped to: {crop_bgr.shape[1]}x{crop_bgr.shape[0]}")

        h, w = crop_bgr.shape[:2]
        expect = 1.46; tol = 0.1
        r_hw, r_wh = h / w, w / h
        if (expect - tol) <= r_hw <= (expect + tol) or (expect - tol) <= r_wh <= (expect + tol):
            # print("Aspect ratio is good, skipping perspective correction")
            warped_bgr = crop_bgr
        else:
            # print(f"Aspect ratio {r_hw:.2f} needs correction (expected ~{expect})")
            warped_bgr, _, _ = warp_card(crop_bgr, debug_view=False)

        if warped_bgr.shape[1] > warped_bgr.shape[0]:
            # print("Rotating card to portrait orientation")
            warped_bgr = cv2.rotate(warped_bgr, cv2.ROTATE_90_CLOCKWISE)

        # print("Testing card orientations (0° and 180°)...")
        candidates = [warped_bgr, cv2.rotate(warped_bgr, cv2.ROTATE_180)]
        best = None; best_score = -1.0
        
        for j, cand in enumerate(candidates):
            angle = "0°" if j == 0 else "180°"
            print(f"Testing orientation {angle}...")
            pil = Image.fromarray(cv2.cvtColor(cand, cv2.COLOR_BGR2RGB))
            res = search_by_image_with_filter(pil, label=label, top_k=top_k, clip_model=_clip, index=_index)
            if res and float(res[0].get("score", 0)) > best_score:
                best, best_score = res[0], float(res[0].get("score", 0))
                # print(f"New best orientation: {angle} (score: {best_score:.4f})")
        
        if best:
            card_id = best.get("card_id")
            ids.append(card_id)
            print(f"Card {i+1} identified: {card_id}")
        else:
            print(f"Card {i+1} could not be identified")
    
    total_time = time.time() - total_start
    print(f"\nDetection completed in {total_time:.2f}s")
    print(f"Final results: {ids}")
    return ids
