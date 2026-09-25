# RAV-21 Semantic Retrieval MVP

RAV-21 biến video camera hành trình thành metadata ROAD++-inspired, tạo `searchable_text`, sinh Gemini embedding và lập chỉ mục FAISS để truy hồi scene theo ngữ nghĩa.

```text
video → Gemini VLM → semantic metadata → searchable_text
      → Gemini Embedding 2 → vector 768D → FAISS IndexFlatIP → scene_id
```

Metadata như agents, actions, locations, weather và events vẫn được lưu riêng để dùng cho metadata filter hoặc hybrid retrieval. Chỉ `searchable_text` được gửi tới embedding API; pipeline không embed JSON, đường dẫn file, timestamp hay provenance.

## Thành phần chính

- `pipeline/retrieval/gemini_vlm.py`: phân tích video và sinh structured metadata.
- `pipeline/retrieval/embedder.py`: Gemini Embedding 2, retry, validation và cache/resume.
- `pipeline/retrieval/faiss_index.py`: kiểm tra artifact và xây `IndexFlatIP`.
- `pipeline/retrieval/vector_search.py`: embed query, tìm top-K và map FAISS ID về `scene_id`.
- `scripts/run_semantic_pipeline.py`: xử lý video theo scene.
- `scripts/build_embeddings.py`: tạo/cập nhật embedding corpus.
- `scripts/build_faiss_index.py`: xây FAISS index.
- `scripts/search.py`: tìm kiếm vector từ command line.

## Cài đặt

Yêu cầu Python 3.10 trở lên.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Project sử dụng Google GenAI SDK mới:

```powershell
pip install google-genai
```

Điền API key vào `.env`:

```dotenv
GEMINI_API_KEY=YOUR_KEY

EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=gemini-embedding-2
EMBEDDING_DIMENSION=768
EMBEDDING_NORMALIZE=true
EMBEDDING_BATCH_SIZE=16
```

Hoặc đặt key trong PowerShell cho session hiện tại:

```powershell
$env:GEMINI_API_KEY="YOUR_KEY"
```

Không commit `.env`. File này đã nằm trong `.gitignore`.

## Cấu hình

`config/retrieval.yaml` chứa cấu hình mặc định:

```yaml
embedding:
  provider: gemini
  model: gemini-embedding-2
  dimension: 768
  normalize: true

index:
  type: IndexFlatIP
  dimension: 768
  version: v1
```

Độ ưu tiên là:

```text
CLI argument > environment/.env > retrieval.yaml > code default
```

Gemini Embedding 2 được gọi với:

```python
types.EmbedContentConfig(output_dimensionality=768)
```

Adapter kiểm tra số chiều, kiểu số, NaN/Inf và norm của mọi vector. Cả document và query dùng cùng model, dimension và normalization policy.

## Dữ liệu đầu vào

Video được đặt theo scene:

```text
data/
└── scene-0061/
    └── cam_front.mp4
```

Semantic corpus được tạo tại:

```text
artifacts/semantic/semantic_corpus.jsonl
```

Mỗi dòng cần tối thiểu:

```json
{
  "scene_id": "scene-0061",
  "searchable_text": "Urban intersection. A pedestrian crosses while a car turns right.",
  "agents": ["pedestrian", "car"],
  "actions": ["crossing", "turning_right"]
}
```

## Chạy pipeline

### 1. Tạo semantic metadata

```powershell
python .\scripts\run_semantic_pipeline.py --scene-id scene-0061
```

Xử lý tất cả scene:

```powershell
python .\scripts\run_semantic_pipeline.py --all
```

Tạo lại artifact đã tồn tại:

```powershell
python .\scripts\run_semantic_pipeline.py --scene-id scene-0061 --force
```

### 2. Tạo Gemini embedding

```powershell
python .\scripts\build_embeddings.py
```

Tạo lại toàn bộ embedding:

```powershell
python .\scripts\build_embeddings.py --force
```

Override bằng CLI:

```powershell
python .\scripts\build_embeddings.py `
  --model gemini-embedding-2 `
  --dimension 768 `
  --batch-size 16
```

Pipeline hỗ trợ resume. Một scene được bỏ qua khi các giá trị sau không đổi:

- `scene_id`;
- `embedding_model`;
- `embedding_dimension`;
- normalization policy;
- embedding pipeline version;
- SHA-256 của `searchable_text`.

Nếu `searchable_text` thay đổi, chỉ scene đó được embed lại. Sau mỗi batch thành công, checkpoint Parquet được cập nhật.

### 3. Xây FAISS index

```powershell
python .\scripts\build_faiss_index.py
```

Builder kiểm tra schema Parquet, model, dimension, scene order, NaN/Inf và ID mapping trước khi tạo:

```python
faiss.IndexFlatIP(768)
```

Vector được chuyển sang `float32` và L2-normalize nhất quán trước khi add vào index.

### 4. Search

```powershell
python .\scripts\search.py `
  "pedestrian crossing while car turns right" `
  --top-k 10
```

Kết quả:

```json
[
  {
    "faiss_id": 1,
    "scene_id": "scene-0061",
    "score": 0.82
  }
]
```

Search đọc model, dimension và normalization policy từ `manifest.json`. Query bị từ chối nếu cấu hình không khớp index.

## Artifact contract

```text
artifacts/
├── semantic/
│   ├── <scene_id>/semantic_metadata.json
│   └── semantic_corpus.jsonl
├── embeddings/
│   ├── embeddings.parquet
│   └── id_map.parquet
└── index/v1/
    ├── index.faiss
    ├── id_map.parquet
    └── manifest.json
```

`embeddings.parquet` chứa:

| Field | Giá trị |
|---|---|
| `scene_id` | ID xuyên suốt pipeline |
| `embedding` | List 768 số `float32` |
| `embedding_model` | `gemini-embedding-2` |
| `embedding_dimension` | `768` |
| `normalized` | `true` |
| `pipeline_version` | `embedding-pipeline-v2` |
| `text_hash` | SHA-256 của `searchable_text` |

`id_map.parquet` chứa đúng mapping:

| faiss_id | scene_id |
|---:|---|
| 0 | scene-0001 |
| 1 | scene-0061 |

`manifest.json` ghi model, dimension, normalization, index type, số scene và các version liên quan.

## Retry và lỗi dữ liệu

Embedding adapter retry có exponential backoff cho lỗi kết nối và HTTP `408`, `429`, `500`, `502`, `503`, `504`. Các lỗi xác thực hoặc cấu hình dừng ngay.

Pipeline không tạo zero vector hoặc tiếp tục âm thầm khi gặp:

- thiếu `GEMINI_API_KEY`;
- `searchable_text` rỗng;
- response không có embedding;
- số embedding trả về không khớp batch;
- vector không phải 768 chiều;
- vector chứa NaN/Inf hoặc có norm bằng 0;
- schema Parquet cũ hoặc không tương thích;
- dimension của FAISS index không khớp manifest;
- document và query dùng model hoặc dimension khác nhau.

## Kiểm thử

Unit test dùng Gemini client mock và không gọi API thật:

```powershell
python -m pytest -q -p no:cacheprovider
```

Test bao phủ adapter 768D, API retry, missing key, input `searchable_text`, `float32`, resume cache, FAISS dimension, manifest và mapping kết quả về `scene_id`.

Smoke test thật (có gọi API và có thể phát sinh chi phí):

```powershell
python .\scripts\smoke_test_embeddings.py
```

## Quyền riêng tư

- VLM upload video lên Gemini Files API.
- Embedding pipeline gửi `searchable_text` lên Gemini Embedding API.
- Không gửi raw semantic JSON, file path, URI, timestamp hoặc provenance vào embedding API.
- Chỉ xử lý dữ liệu được phép gửi tới Gemini.
- Không ghi API key vào source, README, log hoặc artifact.

## Giới hạn hiện tại

- Vector search CLI chưa kết hợp metadata filter thành một hybrid ranking command duy nhất.
- Scene được xử lý tuần tự ở semantic pipeline.
- Chưa có benchmark định lượng cho retrieval quality.
- Cache Gemini Files API chưa được chia sẻ giữa các lần chạy VLM.

Thông tin truy cập dataset trên Google Cloud nằm trong [ggcloud_readme.md](ggcloud_readme.md).
