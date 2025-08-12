### 🧸 功能重點：

* `add(...)`：存進一張圖片（來自 camera topic），加上 step index、episode id 等資料
* `sample_pairs(...)`：從目前 buffer 裡抽兩張圖片，給 VLM 比較誰比較整齊～

---
* 未來你只要這樣用就好：
  ```python
  image_buffer.add(ep_id, step, image, {"obs": obs})
  pair = image_buffer.sample_pairs(1)[0]
  ```

---


> `PreferenceDataset`（收集哪張圖比較整齊）
> `reward_model.py`（小 NN 架構，支援從偏好資料學 reward）？

