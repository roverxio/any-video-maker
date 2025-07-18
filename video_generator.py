#!/usr/bin/env python3
"""
LLM-Powered Video Generator
Main CLI interface for generating videos from text prompts
"""

import argparse
import sys
import os
import logging
from datetime import datetime
from pathlib import Path
import json

from utils.logger import setup_logging
from utils.config import load_config
from utils.folder_manager import create_run_folder
from user_intent_classifier import classify_intent
from product_ad_script_generator import ScriptGenerator as ProductAdScriptGenerator
from story_ad_script_generator import StoryAdGenerator as StoryAdScriptGenerator
from generic_script_generator import GenericScriptGenerator as GenericScriptGenerator
from script_editor import ScriptEditor
from media_generator import MediaGenerator
from video_stitcher import VideoStitcher


def main():
    """Main entry point for the video generator CLI"""
    parser = argparse.ArgumentParser(
        description="Generate videos from text prompts using LLM and AI media generation"
    )
    parser.add_argument(
        "prompt", 
        type=str, 
        help="Text prompt describing the video you want to create"
    )
    parser.add_argument(
        "--reference-url", 
        type=str, 
        help="Optional URL to reference image/video for consistency",
        default=None
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        help="Base output directory (default: ./outputs)",
        default="./outputs"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config()
        
        # Create run folder
        run_folder = create_run_folder(args.output_dir)
        
        # Setup logging
        log_level = "DEBUG" if args.debug else config.get("LOG_LEVEL", "INFO")
        logger = setup_logging(run_folder, log_level)
        logger.info(f"Starting video generation run in: {run_folder}")
        logger.info(f"User prompt: {args.prompt}")
        if args.reference_url:
            logger.info(f"Reference URL: {args.reference_url}")
        
        # Step 1: Classify user intent
        logger.info("Step 1: Classifying user intent...")
        intent = classify_intent(args.prompt, config.get("OPENAI_API_KEY"))
        logger.info(f"Classified intent: {intent}")

        # Step 2: Generate script based on intent
        logger.info(f"Step 2: Generating {intent.get('video_category', 'Everything else')} script...")

        if intent.get('video_category') == "Product Ad":
            script_generator = ProductAdScriptGenerator(config, logger)
        elif intent.get('video_category') == "Story Ad":
            script_generator = StoryAdScriptGenerator(config, logger)
        else: # Everything else
            script_generator = GenericScriptGenerator(config, logger)
            
        script_data = script_generator.generate_script(args.prompt, args.reference_url)
        
        # Step 3: Validate the script
        if script_data:  # Check if the script is valid
            logger.info("Script generated successfully.")
            
            # Save the script to a JSON file
            script_path = run_folder / 'script.json'
            with open(script_path, 'w') as f:
                json.dump(script_data, f, indent=4)
            logger.info(f"Script saved to: {script_path}")

            # User confirmation and editing loop
            while True:
                print("\n" + "="*50)
                print("✨ Script Generation Complete ✨")
                print("="*50)
                print(f"\nDetected Video Category: {intent.get('video_category', 'Everything else')}\n")
                print("--- Generated Script Summary ---")
                try:
                    print(f"Primary Message: {script_data.get('content_strategy', {}).get('primary_message', 'N/A')}")
                    print(f"Target Emotion: {script_data.get('content_strategy', {}).get('target_emotion', 'N/A')}")
                    print(f"Number of Scenes: {len(script_data.get('scenes', []))}")
                    print(f"For the full script, see: {script_path}")
                except Exception as e:
                    logger.warning(f"Could not display script summary: {e}")
                    print("Could not display script summary. Please check the script.json file.")
                print("------------------------------\n")

                proceed = input("Do you want to proceed with media generation? (y/n/edit): ").strip().lower()

                if proceed in ['y', 'yes']:
                    break
                elif proceed in ['n', 'no']:
                    logger.info("User aborted the process after script review.")
                    print("❌ Process aborted by user.")
                    sys.exit(0)
                elif proceed in ['e', 'edit']:
                    edit_prompt = input("Please describe the changes you would like to make: ").strip()
                    if not edit_prompt:
                        print("⚠️ Edit instructions cannot be empty. Please try again.")
                        continue
                    
                    logger.info(f"User is editing the script with prompt: '{edit_prompt}'")
                    editor = ScriptEditor(config, logger)
                    edited_script = editor.edit_script(script_data, edit_prompt)

                    if edited_script and edited_script != script_data:
                        script_data = edited_script
                        logger.info("Script updated successfully after user edit.")
                        # Save the updated script
                        with open(script_path, 'w') as f:
                            json.dump(script_data, f, indent=4)
                        logger.info(f"Updated script saved to: {script_path}")
                        print("✅ Script updated based on your feedback.")
                    else:
                        logger.warning("Script editing failed or produced no changes. Using previous script.")
                        print("⚠️ Could not apply edits. The script remains unchanged.")
                else:
                    print("Invalid input. Please enter 'y', 'n', or 'edit'.")

            # Step 4: Select image generation API
            image_model = input("Choose image generation API: [1] Quality (OpenAI), [2] Speed (Kontext), or [3] Replicate: ")
            logger.info(f"Selected image model: {image_model}")
            
            # Step 5: Generate media
            media_generator = MediaGenerator(config, logger, run_folder)
            media_files = media_generator.generate_all_media(script_data, args.reference_url)
            
            # Step 6: Stitch video
            video_stitcher = VideoStitcher(config, logger, run_folder)
            final_video_path = video_stitcher.stitch_videos(script_data, media_files)
            
            logger.info(f"✅ Video generation completed successfully!")
            logger.info(f"Final video: {final_video_path}")
            logger.info(f"All files saved in: {run_folder}")
            
            print(f"\n🎬 Video generated successfully!")
            print(f"📂 Output folder: {run_folder}")
            print(f"🎥 Final video: {final_video_path}")
            
        else:
            logger.error("Failed to generate a valid script.")
        
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        error_msg = f"❌ Error during video generation: {str(e)}"
        if 'logger' in locals():
            logger.error(error_msg, exc_info=True)
        else:
            print(error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main() 