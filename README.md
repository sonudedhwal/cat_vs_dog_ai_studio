# Cat vs Dog AI Studio

Streamlit + TensorFlow (ML) + Claude (LLM / generative AI) + MySQL.

## Setup
1. `pip install -r requirements.txt`
2. Start **MySQL** in XAMPP (database `cat_dog_db` and table are created automatically; `schema.sql` is optional).
3. (Optional) `set ANTHROPIC_API_KEY=your_key` (Windows) / `export ANTHROPIC_API_KEY=your_key` (Linux/Mac) or paste it in the sidebar.
4. `streamlit run app.py`

Works immediately with a pretrained MobileNetV2. To train your own:
```
data/train/cat/*.jpg
data/train/dog/*.jpg
python train.py --data data --epochs 5      # creates cat_dog_model.keras
```
