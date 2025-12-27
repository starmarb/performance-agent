# PERFORMANCE AGENT

Instagram Content Performance Analysis Agent <!-- for Amorepacific LANEIGE -->

## Project Structure

```
performance_agent/
├── agent.py              # Main agent code
├── data/
│   └── raw/              # Mock Instagram Graph API data
│       ├── api_mock_laneige_jp.json
│       ├── api_mock_laneige_th.json
│       └── api_mock_laneige_sg.json
├── images/               # Sample LANEIGE images
│   ├── 1__제품_단체샷.jpg
│   ├── 2__제품_단독샷.jpg
│   ├── 3__제품_질감샷.jpg
│   ├── 4__제품_및_모델샷.jpg
│   ├── 5__제품없는_모델샷.jpg
│   └── 6__제품_모델_단체샷.jpg
└── analysis_results.json # Output from agent
```

## How to Run

### Demo Mode (uses mock data)
```bash
cd performance_agent
python agent.py demo
```

### Production Mode (when ready)
```bash
# Set your API keys
export INSTAGRAM_TOKEN="your_instagram_access_token"
export CLAUDE_API_KEY="your_anthropic_api_key"

# Run in production mode
python agent.py production
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA SOURCE                              │
│  FileDataLoader (demo) ←→ InstagramAPILoader (production)   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    ANALYZERS                                │
│  MockAnalyzer (demo) ←→ ClaudeAnalyzer (production)         │
│  • Vision Analysis                                          │
│  • Caption Analysis                                         │
│  • Comment Sentiment Analysis                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 METRICS CALCULATOR                          │
│  • Engagement Rate                                          │
│  • Conversion Score                                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 INSIGHT GENERATOR                           │
│  • Per-Account Insights                                     │
│  • Cross-Market Comparison                                  │
│  • Recommendations                                          │
└─────────────────────────────────────────────────────────────┘
```

## Switching from Demo to Production

When the Instagram Graph API is ready, change the following line to connect to the API:

```python
# Demo mode (current)
agent = create_agent("demo", data_dir="data/raw")

# Production mode (when ready)
agent = create_agent(
    "production",
    instagram_token="YOUR_INSTAGRAM_TOKEN",
    claude_api_key="YOUR_CLAUDE_API_KEY"
)
```

## Output

The agent produces a JSON file with:
- Per-post analysis (vision, caption, comments, metrics)
- Per-account insights (top content types, recommendations)
- Cross-market comparison (content preference by country)

## Content Types

| Code | Korean | English |
|------|--------|---------|
| product_group | 제품 단체샷 | Product Group Shot |
| product_solo | 제품 단독샷 | Product Solo Shot |
| product_texture | 제품 질감샷 | Product Texture Shot |
| product_model | 제품 및 모델샷 | Product + Model Shot |
| model_only | 제품없는 모델샷 | Model Only Shot |
| product_models_group | 제품 모델 단체샷 | Product + Multiple Models |
