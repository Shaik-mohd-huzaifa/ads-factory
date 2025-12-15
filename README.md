# AdaptiveBrand Studio

> **One Campaign, Every Platform—Instantly Consistent**

Built for the **Sketch & Search Hackathon** (Google DeepMind × Qdrant × Freepik)

## 🎯 The Problem

Marketing teams waste **15-20 hours per campaign** manually resizing and adapting creatives for different social media platforms, resulting in inconsistent branding and missed deadlines.

## 💡 The Solution

A platform where brands:
1. Upload **brand guidelines** (logos, colors, fonts)
2. Create **one master creative** concept
3. **Auto-generate** variants for all social media formats
4. **Maintain perfect consistency** across all platforms
5. **Ensure brand safety** with built-in guardrails

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Google Gemini** | Campaign briefs, copy generation, brand analysis |
| **Freepik API** | Image generation, format adaptation |
| **Qdrant** | Brand asset storage, similarity search, style consistency |
| **FastAPI** | Backend API |
| **React + Vite** | Frontend UI |
| **TailwindCSS** | Styling |

## 📁 Project Structure

```
├── backend/
│   ├── main.py           # FastAPI application
│   ├── requirements.txt  # Python dependencies
│   └── .env.example      # Environment variables template
├── frontend/
│   ├── src/
│   │   ├── App.jsx       # Main React component
│   │   └── index.css     # TailwindCSS
│   └── package.json
└── README.md
```

## 🚀 Getting Started

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the server
python main.py
```

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies (use Node 20+)
nvm use 20
npm install

# Run dev server
npm run dev
```

### 3. Environment Variables

Create a `.env` file in the backend directory:

```env
GOOGLE_API_KEY=your_google_api_key
FREEPIK_API_KEY=your_freepik_api_key
QDRANT_URL=http://localhost:6333
```

## 🎨 Features

### Brand Kit Management
- Create and manage multiple brand kits
- Define primary colors, fonts, and tone
- Upload brand assets (logos, images)

### Campaign Generation
- AI-powered campaign brief generation
- Multi-platform creative generation
- Support for 7 social media formats:
  - Instagram Feed (1:1)
  - Instagram Story (9:16)
  - Facebook Feed (1.91:1)
  - YouTube Thumbnail (16:9)
  - LinkedIn (1.91:1)
  - Twitter/X (16:9)
  - TikTok (9:16)

### Brand Compliance
- Automatic brand consistency checks
- Copyright safety validation
- Color and style verification

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/brands` | GET, POST | List/create brands |
| `/api/brands/{id}` | GET | Get brand details |
| `/api/brands/{id}/assets` | POST | Upload brand asset |
| `/api/campaigns/generate` | POST | Generate campaign |
| `/api/campaigns` | GET | List campaigns |
| `/api/compliance/check` | POST | Check brand compliance |
| `/api/generate/image` | POST | Generate image |
| `/api/generate/copy` | POST | Generate copy |

## 🏆 Hackathon Judging Criteria

| Criteria | Implementation |
|----------|----------------|
| **Creative Quality** | Professional multi-format campaigns with visual coherence |
| **Search & Similarity** | Qdrant-based brand library, asset search, style matching |
| **Guardrails** | Built-in brand compliance, copyright checking |
| **UX & Tradeoffs** | Clear speed/quality controls, platform selection |
| **Real-World Fit** | Saves marketing teams 15-20 hours per campaign |

## 📜 License

MIT License - Built for Sketch & Search Hackathon 2024
