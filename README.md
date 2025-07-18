# Any Video Maker - CrewAI Edition

A sophisticated video generation pipeline powered by CrewAI, featuring specialized AI agents that work together to create complete video content from user prompts and reference images. The system generates scripts, selects voices, and produces final videos using fal.ai APIs.

## 🚀 Features

- **CrewAI-Powered Workflow**: Three specialized agents working in collaboration with direct media generation
- **Complete Video Production**: From script to final video with all media assets
- **AI Media Generation**: Images, videos, and audio using fal.ai APIs
- **Voice Selection**: Intelligent voice casting using ElevenLabs via fal.ai
- **Video Stitching**: Professional video assembly using ffmpeg
- **Comprehensive Logging**: Detailed workflow tracking and error handling

## 🤖 Agent Architecture

### 1. Video Content Analyst & Summarizer Agent
- **Role**: Analyzes user input (image + text) to create comprehensive video summaries
- **Expertise**: Visual storytelling, cinematography, audience engagement
- **Output**: Visual scenes, voiceover scripts, video configuration

### 2. Video Production Script Engineer Agent
- **Role**: Transforms summaries into production-ready JSON scripts
- **Expertise**: Automated video generation, AI prompt optimization
- **Output**: Structured JSON with detailed prompts and timing

### 3. Voice Casting Director & ElevenLabs Specialist Agent
- **Role**: Selects the most appropriate voice actor from ElevenLabs voice library
- **Expertise**: Voice casting, audience psychology, ElevenLabs voice library
- **Output**: Voice selection with specific voice ID and optimization recommendations

### 4. Direct Media Generation (Non-Agent)
- **Role**: Programmatic execution of media generation using existing modules
- **Expertise**: fal.ai APIs, video generation, audio synthesis, post-production
- **Output**: Complete video with all media assets and final video file

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- `crewai` - AI agent orchestration framework
- `openai` - OpenAI API integration
- `python-dotenv` - Environment variable management
- `fal-client` - fal.ai API integration
- `requests` - HTTP requests
- `moviepy` - Video editing and processing
- `ffmpeg-python` - FFmpeg integration

## 🔧 Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd any-video-maker
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg** (required for video processing):
   - **Windows**: Download from https://ffmpeg.org/download.html
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   FAL_AI_API_KEY=your_fal_ai_api_key_here
   ```

## 🎬 Usage

### Basic Usage
```bash
python main.py "Create a 30-second product showcase for luxury sunglasses"
```

### With Reference Image
```bash
python main.py "Create a 30-second product showcase for luxury sunglasses" --reference-url "https://example.com/sunglasses.jpg"
```

### Verbose Logging
```bash
python main.py "Create a 30-second product showcase for luxury sunglasses" --verbose
```

## 📁 Output Structure

The system generates timestamped output folders with the following structure:

```
outputs/
└── video_gen_DDMM_HHMMSS/
    ├── generation.log
    ├── run_report.json
    ├── complete_workflow.json
    ├── video_scripts/
    │   ├── summary.md
    │   ├── config.json
    │   ├── voiceover.json
    │   ├── script.json
    │   ├── voice_selection.json
    │   └── media_generation.json
    ├── media/
    │   ├── images/
    │   ├── videos/
    │   └── audio/
    ├── temp/
    └── final_video.mp4
```

### Output Files

- **`summary.md`**: Human-readable video summary with scene descriptions
- **`config.json`**: Video configuration (duration, aspect ratio, style)
- **`voiceover.json`**: Voiceover script and scene breakdown
- **`script.json`**: Production-ready JSON script for AI video generation
- **`voice_selection.json`**: Voice actor selection with ElevenLabs voice ID
- **`media_generation.json`**: Media generation results and statistics
- **`complete_workflow.json`**: Complete workflow results
- **`run_report.json`**: Execution summary and status
- **`final_video.mp4`**: The complete generated video

## 🔄 Workflow

1. **User Input**: Text prompt + optional reference image URL
2. **Summary Agent**: Analyzes input and creates video summary with scenes and voiceover
3. **Script Agent**: Transforms summary into structured JSON for production
4. **Voice Agent**: Selects optimal voice actor from ElevenLabs library
5. **Media Generation**: Direct execution generates all media assets (images, videos, audio) using fal.ai
6. **Video Assembly**: Stitches all media together into final video using ffmpeg
7. **Output**: Complete video file with all assets and documentation

## 🎯 Agent Capabilities

### Summary Generator Agent
- Analyzes visual elements and text prompts
- Creates cinematic scene descriptions with camera work
- Generates natural, conversational voiceover scripts
- Determines optimal video configuration
- Ensures technical feasibility for AI generation

### Script Generator Agent
- Converts creative summaries into production-ready structures
- Optimizes prompts for AI image and video generation
- Ensures technical accuracy in timing and composition
- Creates detailed action prompts for motion and animation
- Validates production feasibility

### Voice Selection Agent
- Analyzes video content to determine optimal voice characteristics
- Matches voice qualities to content type and target audience
- Navigates ElevenLabs voice library to find specific voice IDs
- Considers factors like age, gender, accent, and emotional tone
- Provides voice optimization recommendations for production

### Media Generation Handler
- Directly executes media generation without LLM overhead
- Generates high-quality images using fal.ai Flux Pro models
- Creates dynamic videos from static images with motion and animation
- Produces natural-sounding voiceovers using ElevenLabs via fal.ai
- Orchestrates complex media generation pipelines
- Handles post-production tasks like video stitching and audio mixing

## 🛠️ Technical Details

### CrewAI Integration
- **Hybrid Workflow**: Summary → Script → Voice (CrewAI agents) → Direct Media Generation
- **Enhanced Error Handling**: CrewAI's built-in error management
- **Agent Communication**: Structured data passing between agents
- **Context Preservation**: Original user input available to all agents

### Media Generation Pipeline
- **fal.ai Integration**: Uses Flux Pro for images and videos
- **ElevenLabs Voice**: Voice generation via fal.ai endpoints
- **FFmpeg Processing**: Professional video stitching and editing
- **Concurrent Generation**: Parallel processing for efficiency
- **Error Recovery**: Graceful handling of generation failures

### JSON Script Structure
```json
{
  "video_config": {
    "total_duration": 30,
    "aspect_ratio": "9:16",
    "style": "commercial",
    "pacing": "medium",
    "content_type": "advertisement",
    "needs_voiceover": true,
    "needs_background_music": true,
    "voice_characteristics": "Professional, warm, engaging",
    "music_description": "Upbeat, modern, energetic",
    "voice_id": "elevenlabs_voice_id_here"
  },
  "scenes": [
    {
      "scene_id": "scene_1",
      "timestamp": "0-5",
      "duration": 5,
      "scene_composition": "single_shot",
      "voiceover_text": "Introducing the latest in luxury eyewear...",
      "has_reference": true,
      "image_prompt": "Professional product shot of luxury sunglasses...",
      "action_prompt": "Smooth camera movement around the product..."
    }
  ]
}
```

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Key Missing**:
   - Ensure `.env` file exists with `OPENAI_API_KEY`
   - Verify API key is valid and has sufficient credits

2. **FAL AI API Key Missing**:
   - Ensure `.env` file exists with `FAL_AI_API_KEY`
   - Verify API key is valid and has sufficient credits

3. **FFmpeg Not Found**:
   - Install FFmpeg and ensure it's in your system PATH
   - Verify installation with `ffmpeg -version`

4. **JSON Parsing Errors**:
   - Check agent outputs in logs
   - Verify OpenAI API responses are valid JSON

5. **Media Generation Failures**:
   - Check fal.ai API quotas and limits
   - Verify internet connectivity
   - Review error logs for specific failure reasons

### Debug Mode
Use the `--verbose` flag for detailed logging:
```bash
python main.py "your prompt" --verbose
```

## 🔮 Future Enhancements

- **Additional Agents**: Music selection, visual effects, quality assurance
- **Parallel Processing**: Multiple agents working simultaneously
- **Custom Tools**: Integration with additional video generation APIs
- **Template System**: Pre-defined video templates and styles
- **Batch Processing**: Multiple video generation in sequence
- **Quality Control**: Automated video quality assessment
- **Social Media Optimization**: Platform-specific video formatting

## 📄 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here] 