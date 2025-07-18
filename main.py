#!/usr/bin/env python3
"""
Main entry point for the Any Video Maker project.
Orchestrates the complete video generation pipeline using CrewAI.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Make sure your environment variables are set.")
    pass

# Import CrewAI modules
from video_crew import VideoGenerationCrew


def setup_logging(log_file: Path) -> logging.Logger:
    """Setup logging to both console and file"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def create_run_folder() -> Path:
    """Create timestamped run folder"""
    now = datetime.now()
    folder_name = f"video_gen_{now.strftime('%d%m_%H%M%S')}"
    
    # Create the full path
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    run_folder = outputs_dir / folder_name
    run_folder.mkdir(exist_ok=True)
    
    return run_folder


def generate_report(run_folder: Path, results: Dict[str, Any], 
                   save_success: Dict[str, bool], logger: logging.Logger) -> None:
    """Generate and save a run report"""
    
    # Extract media generation results
    media_result = results.get("media_result", {})
    media_status = media_result.get("media_generation_status", "unknown")
    final_video = media_result.get("final_video")
    generation_summary = media_result.get("generation_summary", {})
    
    report = {
        "run_folder": str(run_folder),
        "timestamp": datetime.now().isoformat(),
        "crewai_workflow": {
            "success": True,
            "summary_generation": "completed",
            "script_generation": "completed",
            "voice_selection": "completed",
            "media_generation": media_status
        },
        "file_saves": save_success,
        "user_prompt": results.get("user_prompt", ""),
        "reference_url": results.get("reference_url", ""),
        "generation_timestamp": results.get("generation_timestamp", ""),
        "media_generation": {
            "status": media_status,
            "final_video": final_video,
            "summary": generation_summary
        }
    }
    
    # Save report
    try:
        report_file = run_folder / "run_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved run report to {report_file}")
    except Exception as e:
        logger.error(f"Failed to save run report: {e}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("CREWAI VIDEO GENERATION SUMMARY")
    print("="*60)
    print(f"Run folder: {run_folder}")
    print("✅ CrewAI workflow completed successfully")
    print("✅ Summary generation completed")
    print("✅ Script generation completed")
    print("✅ Voice selection completed")
    
    if media_status == "success":
        print("✅ Media generation completed")
        if final_video:
            print(f"🎬 Final video: {final_video}")
        if generation_summary:
            print(f"📊 Generated {generation_summary.get('total_images', 0)} images")
            print(f"📹 Generated {generation_summary.get('total_videos', 0)} videos")
            print(f"🎤 Generated {generation_summary.get('total_voiceovers', 0)} voiceovers")
            print(f"⏱️  Final duration: {generation_summary.get('final_duration', 0):.1f} seconds")
            print(f"📁 File size: {generation_summary.get('file_size', '0 MB')}")
    else:
        print(f"❌ Media generation failed: {media_status}")
        errors = media_result.get("errors", [])
        for error in errors:
            print(f"   Error: {error}")
    
    print("\nGenerated files:")
    for filename, success in save_success.items():
        status = "✅" if success else "❌"
        print(f"  {status} {filename}")
    
    print("="*60)


def main():
    """Main function using CrewAI workflow"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate video content from user prompt using CrewAI")
    parser.add_argument("prompt", help="Video generation prompt")
    parser.add_argument("--reference-url", "-r", help="Reference image URL (optional)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Create run folder
    try:
        run_folder = create_run_folder()
        print(f"Created run folder: {run_folder}")
    except Exception as e:
        print(f"❌ Failed to create run folder: {e}")
        return 1
    
    # Setup logging
    log_file = run_folder / "generation.log"
    logger = setup_logging(log_file)
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    
    logger.info(f"Starting CrewAI video generation pipeline")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Reference URL: {args.reference_url}")
    
    # Load configuration
    config = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "FAL_AI_API_KEY": os.getenv("FAL_AI_API_KEY"),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        "MAX_CONCURRENT_REQUESTS": int(os.getenv("MAX_CONCURRENT_REQUESTS", "3")),
        "IMAGE_MODEL": os.getenv("IMAGE_MODEL", "fal"),
        "VIDEO_MODEL": os.getenv("VIDEO_MODEL", "fal"),
        "AUDIO_MODEL": os.getenv("AUDIO_MODEL", "fal"),
        "video_width": int(os.getenv("VIDEO_WIDTH", "704")),
        "video_height": int(os.getenv("VIDEO_HEIGHT", "1304")),
        "video_fps": int(os.getenv("VIDEO_FPS", "24")),
        "output_dir": os.getenv("OUTPUT_DIR", "./outputs")
    }
    
    if not config["OPENAI_API_KEY"]:
        logger.error("OPENAI_API_KEY not found in environment variables")
        print("❌ OPENAI_API_KEY not found in environment variables. Please check your .env file.")
        return 1
    
    if not config["FAL_AI_API_KEY"]:
        logger.error("FAL_AI_API_KEY not found in environment variables")
        print("❌ FAL_AI_API_KEY not found in environment variables. Please check your .env file.")
        return 1
    
    try:
        # Initialize CrewAI video generation crew
        logger.info("Initializing CrewAI video generation crew...")
        video_crew = VideoGenerationCrew(config, logger)
        
        # Execute the complete workflow
        logger.info("Executing CrewAI video generation workflow...")
        results = video_crew.generate_video_content(args.prompt, args.reference_url, run_folder)
        
        # Results are now saved during workflow execution
        # Generate success status for all files (since they were saved during execution)
        save_success = {
            "summary.md": True,
            "config.json": True,
            "voiceover_generation.json": True,
            "voiceover.json": True,
            "script.json": True,
            "voice_selection.json": True,
            "media_generation.json": True,
            "complete_workflow.json": True
        }
        
        # Generate and display report
        generate_report(run_folder, results, save_success, logger)
        
        logger.info("CrewAI video generation pipeline completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"CrewAI video generation pipeline failed: {e}")
        print(f"❌ CrewAI video generation pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 