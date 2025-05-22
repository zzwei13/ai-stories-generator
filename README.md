# IR-GenAI_Final_Project - AI-Driven Personalized Storytelling and Content Creation Platform

## 專題說明
本專題致力於開發一個結合 生成式AI 與 自動提示工程（Automatic Prompt Engineering） 的故事創作平台。使用者只需輸入偏好（如主題、語氣、角色等），系統便會根據這些輸入，透過自動化提示優化技術，動態生成具備一致性、創意與個人化的文字內容（例如短篇故事、部落格文章或互動對話）。

系統具備回饋學習機制，不僅能自動評估文本品質，還能根據使用者評分與意見持續優化生成內容。最終目標是提供一個高度互動、個性化且跨模態（文字+圖片）的內容生成體驗。

## 整體流程關鍵技術點
1.RAG + Self-Reflection 應用於 prompt 優化與故事生成
2.多模型語言生成，提升內容的多樣性與準確性
3.使用者回饋與 NLP 分數共同驅動 prompt 再優化
4.結合向量檢索與知識資料庫提升生成內容一致性
5.圖文並茂輸出，創造個性化故事體驗

---

## 技術說明

| Layer            | Technology                                                   |
|------------------|--------------------------------------------------------------|
| Language Model   | Together AI, Hugging Face Transformers                       |
| Prompt Engineering | LangChain, Custom Python Pipelines                         |
| Backend          | Python (Streamlit-based App)                                 |
| Frontend         | Streamlit                                                    |
| Storage          | JSON-based file storage, session logs in text format         |
| Evaluation       | Cosine Similarity, Type-Token Ratio (TTR), User Feedback      |

---
### Set up
- pip install -r requirements.txt
- go to https://www.together.ai/ to get your API key
- create .env file then paste your API key in it.
- E.g. TOGETHER_API_KEY3=1234

### RUN CODE AT CMD
- cd {the folder path of IR_Final.py }
- streamlit run IR_Final.py

### About Code
- IR_Final: UI、Optimize Prompt、Main function(merge all function)、User feedback、set up and so on...

##### 1.User Preference Interface
##### 2.Dynamic Prompt Engineering Module
##### 3.Generative AI Integration
##### 4.Automatic Content Evaluation & Feedback Loop
##### 5.Output Visualization and Personalization

- StoryRAG:Generate Story by midterm database.(Create by RAG)

- nlp_sentenceBert: NLP Computation.
- 
## User Interface

使用 Streamlit 打造互動介面，支援以下功能：  
- 主題、語調、風格、角色等偏好輸入  
- 多模型與 NLP 評分目標選擇  
- 可檢視故事與圖片結果、提供使用者回饋並進一步優化

### 主介面  
![ui01](static/img/ui01.png)

### 使用者回饋介面  
![ui02](static/img/ui.png)

## DEMO

1. [![Demo 1](https://img.youtube.com/vi/SEkmVnCEPo/0.jpg)](https://youtu.be/SEkmVnCEPo)  
2. [![Demo 2](https://img.youtube.com/vi/axzKIcIC_zs/0.jpg)](https://youtu.be/axzKIcIC_zs)  
3. [![Demo 3](https://img.youtube.com/vi/XiT6RNdnPKM/0.jpg)](https://youtu.be/XiT6RNdnPKM)



