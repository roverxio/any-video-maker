#!/usr/bin/env python3
"""
Main entry point for the Any Video Maker project.
Orchestrates the complete video generation pipeline from user prompt to final script.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Import our modules
from video_summary_generator import VideoSummaryGenerator
from video_script_generator import VideoScriptGenerator


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
    
    # Create video_scripts subfolder
    scripts_folder = run_folder / "video_scripts"
    scripts_folder.mkdir(exist_ok=True)
    
    return run_folder


def save_summary_outputs(summary_result: Dict[str, Any], scripts_folder: Path, logger: logging.Logger) -> Dict[str, bool]:
    """Save video summary outputs and return success status for each file"""
    success_status = {}
    
    try:
        # Save visual summary as markdown
        summary_file = scripts_folder / "summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# Video Summary\n\n")
            f.write(summary_result["visual_summary"])
        logger.info(f"Saved visual summary to {summary_file}")
        success_status["summary.md"] = True
    except Exception as e:
        logger.error(f"Failed to save summary.md: {e}")
        success_status["summary.md"] = False
    
    try:
        # Save video config as JSON
        config_file = scripts_folder / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(summary_result.get("video_config", {}), f, indent=2)
        logger.info(f"Saved video config to {config_file}")
        success_status["config.json"] = True
    except Exception as e:
        logger.error(f"Failed to save config.json: {e}")
        success_status["config.json"] = False
    
    try:
        # Save voiceover as JSON
        voiceover_file = scripts_folder / "voiceover.json"
        voiceover_data = {
            "voiceover_text": summary_result["voiceover"],
            "scenes": summary_result["voiceover_scenes"]
        }
        with open(voiceover_file, 'w', encoding='utf-8') as f:
            json.dump(voiceover_data, f, indent=2)
        logger.info(f"Saved voiceover data to {voiceover_file}")
        success_status["voiceover.json"] = True
    except Exception as e:
        logger.error(f"Failed to save voiceover.json: {e}")
        success_status["voiceover.json"] = False
    
    return success_status


def save_final_script(script_result: Dict[str, Any], scripts_folder: Path, logger: logging.Logger) -> bool:
    """Save final script JSON and return success status"""
    try:
        script_file = scripts_folder / "final_script.json"
        with open(script_file, 'w', encoding='utf-8') as f:
            json.dump(script_result, f, indent=2)
        logger.info(f"Saved final script to {script_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to save final_script.json: {e}")
        return False


def generate_report(run_folder: Path, summary_success: Dict[str, bool], 
                   script_success: bool, summary_error: Optional[str], 
                   script_error: Optional[str], logger: logging.Logger) -> None:
    """Generate and save a run report"""
    
    report = {
        "run_folder": str(run_folder),
        "timestamp": datetime.now().isoformat(),
        "summary_generation": {
            "success": summary_error is None,
            "error": summary_error,
            "files": summary_success
        },
        "script_generation": {
            "success": script_success,
            "error": script_error
        },
        "overall_success": summary_error is None and script_success
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
    print("RUN SUMMARY")
    print("="*60)
    print(f"Run folder: {run_folder}")
    
    if summary_error:
        print(f"❌ Video summary generation FAILED: {summary_error}")
    else:
        print("✅ Video summary generation SUCCESS")
        for filename, success in summary_success.items():
            status = "✅" if success else "❌"
            print(f"  {status} {filename}")
    
    if summary_error:
        print("❌ Script generation SKIPPED (summary failed)")
    elif script_success:
        print("✅ Final script generation SUCCESS")
    else:
        print(f"❌ Final script generation FAILED: {script_error}")
    
    print("="*60)


def main():
    """Main function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate video content from user prompt")
    parser.add_argument("prompt", help="Video generation prompt")
    parser.add_argument("--reference-url", "-r", help="Reference image URL (optional)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Create run folder
    try:
        run_folder = create_run_folder()
        scripts_folder = run_folder / "video_scripts"
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
    
    logger.info(f"Starting video generation pipeline")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Reference URL: {args.reference_url}")
    
    # Initialize variables for error tracking
    summary_result = None
    summary_success = {}
    summary_error = None
    script_success = False
    script_error = None
    
    # Step 1: Generate video summary and voiceover
    try:
        logger.info("Step 1: Generating video summary and voiceover...")
        
        # Load config
        config = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
        }
        
        if not config["OPENAI_API_KEY"]:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
        
        # Initialize and run video summary generator
        summary_generator = VideoSummaryGenerator(config, logger)
        summary_result = summary_generator.generate_complete_video_content(
            args.prompt, 
            args.reference_url
        )
        
        logger.info("Video summary generation completed successfully")
        
        # Save summary outputs
        summary_success = save_summary_outputs(summary_result, scripts_folder, logger)
        
    except Exception as e:
        summary_error = str(e)
        logger.error(f"Video summary generation failed: {e}")
    
    # Step 2: Generate final script (only if summary succeeded)
    if summary_result is not None:
        try:
            logger.info("Step 2: Generating final video script...")
            
            # Initialize script generator
            script_generator = VideoScriptGenerator(logger)
            
            # Prepare voiceover data
            voiceover_data = {
                "voiceover_text": summary_result["voiceover"],
                "scenes": summary_result["voiceover_scenes"]
            }
            
            # Extract video config from summary result
            video_config = summary_result.get("video_config", {
                "total_duration": 30,
                "aspect_ratio": "9:16", 
                "video_style": "commercial"
            })
            
            # Generate script
            script_result = script_generator.generate_script(
                visual_summary=summary_result["visual_summary"],
                voiceover_data=voiceover_data,
                video_config=video_config,
                user_prompt=args.prompt,
                reference_url=args.reference_url
            )
            
            # Save final script
            script_success = save_final_script(script_result, scripts_folder, logger)
            
            if script_success:
                logger.info("Final script generation completed successfully")
            
        except Exception as e:
            script_error = str(e)
            logger.error(f"Final script generation failed: {e}")
    else:
        logger.warning("Skipping script generation due to summary failure")
    
    # Generate final report
    generate_report(run_folder, summary_success, script_success, 
                   summary_error, script_error, logger)
    
    # Return appropriate exit code
    if summary_error or not script_success:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main()) 