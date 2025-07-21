"""
CrewAI Crew for Video Generation Pipeline
Orchestrates the workflow between summary and script generation agents
"""

from crewai import Crew
from typing import Dict, Any, Optional
import logging
import json
import os
from pathlib import Path
from datetime import datetime

from agents import VideoAgents
from tasks import VideoTasks
from media_generation_handler import MediaGenerationHandler


class VideoGenerationCrew:
    """Crew that orchestrates video generation workflow"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.agents = VideoAgents(config, logger)
        self.tasks = VideoTasks(logger)
    
    def generate_video_content(self, user_prompt: str, reference_url: Optional[str] = None, 
                             output_folder: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate complete video content using CrewAI workflow
        
        Args:
            user_prompt: Text description of desired video
            reference_url: Optional reference image URL
            output_folder: Optional output folder path
            
        Returns:
            Dictionary containing all results including final video
        """
        
        try:
            self.logger.info("Starting CrewAI video generation workflow")
            
            # Create output folder if not provided
            if output_folder is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_folder = Path(f"outputs/run_{timestamp}")
                output_folder.mkdir(parents=True, exist_ok=True)
            
            # Create scripts subfolder for saving intermediate results
            scripts_folder = output_folder / "video_scripts"
            scripts_folder.mkdir(exist_ok=True)
            
            # Create agents
            summary_agent = self.agents.create_summary_generator_agent()
            voiceover_agent = self.agents.create_voiceover_generator_agent()
            script_agent = self.agents.create_script_generator_agent()
            voice_agent = self.agents.create_voice_selection_agent()
            
            # Step 1: Generate Summary
            summary_task = self.tasks.create_summary_generation_task(
                summary_agent, user_prompt, reference_url
            )
            
            summary_crew = Crew(
                agents=[summary_agent],
                tasks=[summary_task],
                verbose=True,
                memory=False
            )
            
            self.logger.info("Executing summary generation task...")
            summary_result = summary_crew.kickoff()
            summary_data = self._parse_crew_result(summary_result, "summary")
            
            # Save summary results immediately
            self._save_summary_results(summary_data, scripts_folder)
            
            # Step 2: Generate Voiceover
            voiceover_task = self.tasks.create_voiceover_generation_task(
                voiceover_agent, summary_data
            )
            
            voiceover_crew = Crew(
                agents=[voiceover_agent],
                tasks=[voiceover_task],
                verbose=True,
                memory=False
            )
            
            self.logger.info("Executing voiceover generation task...")
            voiceover_result = voiceover_crew.kickoff()
            voiceover_data = self._parse_crew_result(voiceover_result, "voiceover")
            
            # Save voiceover results immediately
            self._save_voiceover_results(voiceover_data, summary_data, scripts_folder)
            
            # Step 3: Generate Script
            script_task = self.tasks.create_script_generation_task(
                script_agent, summary_data, voiceover_data, reference_url
            )
            
            script_crew = Crew(
                agents=[script_agent],
                tasks=[script_task],
                verbose=True,
                memory=False
            )
            
            self.logger.info("Executing script generation task...")
            script_result = script_crew.kickoff()
            final_script = self._parse_crew_result(script_result, "script")
            
            # Save script results immediately
            self._save_script_results(final_script, scripts_folder)
            
            # Step 4: Select Voice
            voice_task = self.tasks.create_voice_selection_task(
                voice_agent, final_script, user_prompt, reference_url
            )
            
            voice_crew = Crew(
                agents=[voice_agent],
                tasks=[voice_task],
                verbose=True,
                memory=False
            )
            
            self.logger.info("Executing voice selection task...")
            voice_result = voice_crew.kickoff()
            voice_data = self._parse_crew_result(voice_result, "voice")
            
            # Save voice selection results immediately
            self._save_voice_results(voice_data, scripts_folder)
            
            # Step 5: Generate Media (using direct execution, not CrewAI agent)
            self.logger.info("Executing media generation...")
            media_handler = MediaGenerationHandler(self.config, self.logger, output_folder)
            media_result = media_handler.generate_media(
                final_script, voice_data, user_prompt, reference_url
            )
            
            # Save media generation results immediately
            self._save_media_results(media_result, scripts_folder)
            
            # Combine all results
            complete_result = {
                "summary_result": summary_data,
                "voiceover_result": voiceover_data,
                "script_result": final_script,
                "voice_result": voice_data,
                "media_result": media_result,
                "user_prompt": user_prompt,
                "reference_url": reference_url,
                "generation_timestamp": datetime.now().isoformat(),
                "output_folder": str(output_folder)
            }
            
            # Save complete workflow results
            self._save_complete_results(complete_result, output_folder)
            
            self.logger.info("CrewAI video generation workflow completed successfully")
            return complete_result
            
        except Exception as e:
            self.logger.error(f"CrewAI workflow failed: {str(e)}")
            raise
    
    def _parse_crew_result(self, result, step_name: str) -> Dict[str, Any]:
        """Parse CrewAI result and extract JSON content"""
        
        try:
            # CrewAI returns a CrewOutput object, we need to access the result
            if hasattr(result, 'raw'):
                content = result.raw
            elif hasattr(result, 'result'):
                content = result.result
            else:
                content = str(result)
            
            # Log the raw content for debugging
            self.logger.debug(f"Raw {step_name} result: {content}")
            
            # Remove markdown code block wrapper if present
            if content.startswith('```json'):
                content = content[7:]  # Remove ```json
            if content.startswith('```'):
                content = content[3:]   # Remove ```
            if content.endswith('```'):
                content = content[:-3]  # Remove closing ```
            
            # Clean up any extra whitespace
            content = content.strip()
            
            # Try to parse the JSON response
            try:
                parsed_data = json.loads(content)
                self.logger.info(f"{step_name.capitalize()} generation completed successfully")
                return parsed_data
            except json.JSONDecodeError as json_error:
                # Try to fix common JSON issues (like trailing commas) with json5
                try:
                    import json5
                    parsed_data = json5.loads(content)
                    self.logger.info(f"{step_name.capitalize()} generation completed successfully (fixed with json5)")
                    return parsed_data
                except (ImportError, Exception) as json5_error:
                    # json5 not available or failed, continue with original error handling
                    pass
                # If JSON parsing fails, try to extract JSON from the response
                self.logger.warning(f"Initial JSON parsing failed for {step_name}: {json_error}")
                
                # Try to find JSON content within the response
                import re
                json_pattern = r'\{.*\}'
                json_matches = re.findall(json_pattern, content, re.DOTALL)
                
                if json_matches:
                    # Try the first JSON match
                    for i, json_match in enumerate(json_matches):
                        try:
                            parsed_data = json.loads(json_match)
                            self.logger.info(f"{step_name.capitalize()} generation completed successfully (extracted JSON #{i+1})")
                            return parsed_data
                        except json.JSONDecodeError:
                            continue
                
                # If all else fails, create a fallback response
                if step_name == "voice":
                    self.logger.warning(f"Creating fallback voice selection for {step_name}")
                    fallback_response = {
                        "voice_selection": {
                            "voice_id": "21m00Tcm4TlvDq8ikWAM",  # Default ElevenLabs voice ID
                            "voice_name": "Rachel",
                            "voice_characteristics": {
                                "age": "Young",
                                "gender": "Female",
                                "accent": "American",
                                "tone": "Friendly",
                                "style": "Conversational"
                            },
                            "selection_reasoning": "Default voice selected due to parsing error",
                            "content_match_score": 7,
                            "target_audience_alignment": "General audience friendly"
                        },
                        "voiceover_optimization": {
                            "recommended_pacing": "medium",
                            "emotional_emphasis": "Natural conversational tone",
                            "technical_notes": "Standard voice generation settings"
                        }
                    }
                    self.logger.info(f"Using fallback voice selection for {step_name}")
                    return fallback_response
                elif step_name == "voiceover":
                    self.logger.warning(f"Creating fallback voiceover generation for {step_name}")
                    fallback_response = {
                        "voiceover_generation": {
                            "detected_style": "narrator",
                            "overall_tone": "friendly and conversational",
                            "full_voiceover_text": "Scene 1: Check this out.\nScene 2: Pretty amazing stuff.\nScene 3: This changes everything.",
                            "scenes": [
                                {
                                    "scene_number": "Scene 1",
                                    "voiceover_text": "Check this out.",
                                    "word_count": 3,
                                    "estimated_duration": 1.2,
                                    "scene_duration": 5,
                                    "timing_match": "good"
                                },
                                {
                                    "scene_number": "Scene 2", 
                                    "voiceover_text": "Pretty amazing stuff.",
                                    "word_count": 3,
                                    "estimated_duration": 1.2,
                                    "scene_duration": 5,
                                    "timing_match": "good"
                                },
                                {
                                    "scene_number": "Scene 3",
                                    "voiceover_text": "This changes everything.",
                                    "word_count": 3,
                                    "estimated_duration": 1.2,
                                    "scene_duration": 5,
                                    "timing_match": "good"
                                }
                            ],
                            "style_justification": "Fallback narrator style selected due to parsing error",
                            "technical_notes": "Standard pacing and emphasis"
                        }
                    }
                    self.logger.info(f"Using fallback voiceover generation for {step_name}")
                    return fallback_response
                
                # For other steps, raise the original error
                raise json_error
            
        except Exception as e:
            self.logger.error(f"Failed to parse {step_name} result: {e}")
            self.logger.error(f"Raw result type: {type(result)}")
            self.logger.error(f"Raw result content: {result}")
            
            # Provide fallbacks for specific steps
            if step_name == "voice":
                self.logger.warning("Providing fallback voice selection due to parsing error")
                return {
                    "voice_selection": {
                        "voice_id": "21m00Tcm4TlvDq8ikWAM",
                        "voice_name": "Rachel",
                        "voice_characteristics": {
                            "age": "Young",
                            "gender": "Female",
                            "accent": "American",
                            "tone": "Friendly",
                            "style": "Conversational"
                        },
                        "selection_reasoning": "Default voice selected due to parsing error",
                        "content_match_score": 7,
                        "target_audience_alignment": "General audience friendly"
                    },
                    "voiceover_optimization": {
                        "recommended_pacing": "medium",
                        "emotional_emphasis": "Natural conversational tone",
                        "technical_notes": "Standard voice generation settings"
                    }
                }
            elif step_name == "voiceover":
                self.logger.warning("Providing fallback voiceover generation due to parsing error")
                return {
                    "voiceover_generation": {
                        "detected_style": "narrator",
                        "overall_tone": "friendly and conversational",
                        "full_voiceover_text": "Scene 1: Check this out.\nScene 2: Pretty amazing stuff.\nScene 3: This changes everything.",
                        "scenes": [
                            {
                                "scene_number": "Scene 1",
                                "voiceover_text": "Check this out.",
                                "word_count": 3,
                                "estimated_duration": 1.2,
                                "scene_duration": 5,
                                "timing_match": "good"
                            },
                            {
                                "scene_number": "Scene 2",
                                "voiceover_text": "Pretty amazing stuff.",
                                "word_count": 3,
                                "estimated_duration": 1.2,
                                "scene_duration": 5,
                                "timing_match": "good"
                            },
                            {
                                "scene_number": "Scene 3",
                                "voiceover_text": "This changes everything.",
                                "word_count": 3,
                                "estimated_duration": 1.2,
                                "scene_duration": 5,
                                "timing_match": "good"
                            }
                        ],
                        "style_justification": "Fallback narrator style selected due to parsing error",
                        "technical_notes": "Standard pacing and emphasis"
                    }
                }
            
            raise ValueError(f"{step_name.capitalize()} agent did not return valid JSON: {str(e)}")
    
    def save_results(self, results: Dict[str, Any], output_folder: Path) -> Dict[str, bool]:
        """
        Save the generated results to files
        
        Args:
            results: Complete results from crew workflow
            output_folder: Path to save results
            
        Returns:
            Dictionary indicating success status for each file
        """
        
        success_status = {}
        
        try:
            # Create scripts subfolder
            scripts_folder = output_folder / "video_scripts"
            scripts_folder.mkdir(exist_ok=True)
            
            # Save summary results
            summary_result = results["summary_result"]
            
            # Save visual summary as markdown
            try:
                summary_file = scripts_folder / "summary.md"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write("# Video Summary\n\n")
                    f.write(summary_result.get("visual_summary", ""))
                self.logger.info(f"Saved visual summary to {summary_file}")
                success_status["summary.md"] = True
            except Exception as e:
                self.logger.error(f"Failed to save summary.md: {e}")
                success_status["summary.md"] = False
            
            # Save video config as JSON
            try:
                config_file = scripts_folder / "config.json"
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(summary_result.get("video_config", {}), f, indent=2)
                self.logger.info(f"Saved video config to {config_file}")
                success_status["config.json"] = True
            except Exception as e:
                self.logger.error(f"Failed to save config.json: {e}")
                success_status["config.json"] = False
            
            # Save dedicated voiceover results
            voiceover_result = results.get("voiceover_result", {})
            try:
                voiceover_file = scripts_folder / "voiceover_generation.json"
                with open(voiceover_file, 'w', encoding='utf-8') as f:
                    json.dump(voiceover_result, f, indent=2)
                self.logger.info(f"Saved voiceover generation to {voiceover_file}")
                success_status["voiceover_generation.json"] = True
            except Exception as e:
                self.logger.error(f"Failed to save voiceover_generation.json: {e}")
                success_status["voiceover_generation.json"] = False
            
            # Save script results
            script_result = results["script_result"]
            
            try:
                script_file = scripts_folder / "script.json"
                with open(script_file, 'w', encoding='utf-8') as f:
                    json.dump(script_result, f, indent=2)
                self.logger.info(f"Saved script to {script_file}")
                success_status["script.json"] = True
            except Exception as e:
                self.logger.error(f"Failed to save script.json: {e}")
                success_status["script.json"] = False
            
            # Save voice selection results
            voice_result = results["voice_result"]
            
            try:
                voice_file = scripts_folder / "voice_selection.json"
                with open(voice_file, 'w', encoding='utf-8') as f:
                    json.dump(voice_result, f, indent=2)
                self.logger.info(f"Saved voice selection to {voice_file}")
                success_status["voice_selection.json"] = True
            except Exception as e:
                self.logger.error(f"Failed to save voice_selection.json: {e}")
                success_status["voice_selection.json"] = False
            
            # Save media generation results
            media_result = results["media_result"]
            
            try:
                media_file = scripts_folder / "media_generation.json"
                with open(media_file, 'w', encoding='utf-8') as f:
                    json.dump(media_result, f, indent=2)
                self.logger.info(f"Saved media generation results to {media_file}")
                success_status["media_generation.json"] = True
            except Exception as e:
                self.logger.error(f"Failed to save media_generation.json: {e}")
                success_status["media_generation.json"] = False
            
            # Save complete workflow results
            try:
                complete_file = output_folder / "complete_workflow.json"
                with open(complete_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
                self.logger.info(f"Saved complete workflow to {complete_file}")
                success_status["complete_workflow.json"] = True
            except Exception as e:
                self.logger.error(f"Failed to save complete_workflow.json: {e}")
                success_status["complete_workflow.json"] = False
            
            return success_status
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            return {"error": False} 

    def _save_summary_results(self, summary_data: Dict[str, Any], scripts_folder: Path) -> None:
        """Save summary results immediately after generation"""
        try:
            # Save visual summary as markdown
            summary_file = scripts_folder / "summary.md"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("# Video Summary\n\n")
                f.write(summary_data.get("visual_summary", ""))
            self.logger.info(f"✅ Saved visual summary to {summary_file}")
            
            # Save video config as JSON
            config_file = scripts_folder / "config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data.get("video_config", {}), f, indent=2)
            self.logger.info(f"✅ Saved video config to {config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save summary results: {e}")
    
    def _save_voiceover_results(self, voiceover_data: Dict[str, Any], summary_data: Dict[str, Any], scripts_folder: Path) -> None:
        """Save voiceover results immediately after generation"""
        try:
            # Save dedicated voiceover results
            voiceover_file = scripts_folder / "voiceover_generation.json"
            with open(voiceover_file, 'w', encoding='utf-8') as f:
                json.dump(voiceover_data, f, indent=2)
            self.logger.info(f"✅ Saved voiceover generation to {voiceover_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save voiceover results: {e}")
    
    def _save_script_results(self, script_data: Dict[str, Any], scripts_folder: Path) -> None:
        """Save script results immediately after generation"""
        try:
            script_file = scripts_folder / "script.json"
            with open(script_file, 'w', encoding='utf-8') as f:
                json.dump(script_data, f, indent=2)
            self.logger.info(f"✅ Saved script to {script_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save script results: {e}")
    
    def _save_voice_results(self, voice_data: Dict[str, Any], scripts_folder: Path) -> None:
        """Save voice selection results immediately after generation"""
        try:
            voice_file = scripts_folder / "voice_selection.json"
            with open(voice_file, 'w', encoding='utf-8') as f:
                json.dump(voice_data, f, indent=2)
            self.logger.info(f"✅ Saved voice selection to {voice_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save voice selection results: {e}")
    
    def _save_media_results(self, media_data: Dict[str, Any], scripts_folder: Path) -> None:
        """Save media generation results immediately after generation"""
        try:
            media_file = scripts_folder / "media_generation.json"
            with open(media_file, 'w', encoding='utf-8') as f:
                json.dump(media_data, f, indent=2)
            self.logger.info(f"✅ Saved media generation results to {media_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save media generation results: {e}")
    
    def _save_complete_results(self, complete_data: Dict[str, Any], output_folder: Path) -> None:
        """Save complete workflow results"""
        try:
            complete_file = output_folder / "complete_workflow.json"
            with open(complete_file, 'w', encoding='utf-8') as f:
                json.dump(complete_data, f, indent=2)
            self.logger.info(f"✅ Saved complete workflow to {complete_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save complete workflow results: {e}") 