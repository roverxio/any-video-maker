# Any Video Maker

A Python tool for generating video content with AI-powered scripts and summaries.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 main.py "create an ad for this coffee machine" --reference-url https://m.media-amazon.com/images/I/61dNcZx6yWL.jpg
```

## Environment Setup

Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

## Output

Generated videos and scripts are saved to the `outputs/` directory. 