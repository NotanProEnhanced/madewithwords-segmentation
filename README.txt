Segmentation Microservice (Render Free Tier) — mask + bbox + confidence

Endpoints:
- GET  /health
- POST /segment  (multipart form-data: image=@file)
  -> JSON {mask_url, bbox, coverage, confidence, mask_id}
- GET  /mask/<mask_id>.png -> PNG mask (255 foreground)

Render settings:
- Build command:  pip install -r requirements.txt
- Start command:  gunicorn -b 0.0.0.0:$PORT app:app

Test:
1) Health:
   https://YOUR-SERVICE.onrender.com/health

2) Segment:
   curl -s -X POST -F "image=@dog.jpg" https://YOUR-SERVICE.onrender.com/segment

Note:
Mask storage is short-lived in memory. Your PHP app should immediately download mask_url
and cache it locally (recommended: /madewithwords/generated/masks/).
