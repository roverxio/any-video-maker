"""
CrewAI Agents for Video Generation Pipeline
Defines specialized agents for video content creation
"""

from crewai import Agent
from typing import Dict, Any, Optional
import logging


class VideoAgents:
    """Container for all video generation agents"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def create_summary_generator_agent(self) -> Agent:
        """Create the Video Summary Generator Agent"""
        
        return Agent(
            role="Video Content Analyst & Summarizer",
            goal="Analyze user input (image + text) to create comprehensive video summaries with visual scenes, voiceover scripts, and production configurations",
            backstory="""You are an expert video content analyst with 15+ years of experience in film production, 
            advertising, and digital content creation. You have a deep understanding of visual storytelling, 
            cinematography, and audience engagement. Your specialty is taking raw creative inputs and transforming 
            them into structured, production-ready video concepts that maintain visual consistency and narrative flow.
            
            You excel at:
            - Analyzing visual elements and text prompts to understand user intent
            - Creating cinematic scene descriptions with proper camera work and lighting
            - Generating natural, conversational voiceover scripts
            - Determining optimal video configurations (duration, aspect ratio, style)
            - Ensuring technical feasibility for AI video generation""",
            
            verbose=True,
            allow_delegation=False,
            tools=[],
            
            # Agent-specific configuration
            config={
                "openai_api_key": self.config["OPENAI_API_KEY"],
                "model": "gpt-4o",
                "temperature": 0.7
            }
        )
    
    def create_script_generator_agent(self) -> Agent:
        """Create the Video Script Generator Agent"""
        
        return Agent(
            role="Video Production Script Engineer",
            goal="Transform video summaries into detailed, structured JSON scripts optimized for automated video production systems",
            backstory="""You are a senior video production engineer with expertise in automated video generation 
            systems and AI-powered content creation. You have worked with major streaming platforms, advertising 
            agencies, and content creation tools. Your role is to bridge the gap between creative concepts and 
            technical implementation.
            
            You specialize in:
            - Converting creative summaries into production-ready JSON structures
            - Optimizing prompts for AI image and video generation systems
            - Ensuring technical accuracy in scene timing and composition
            - Creating detailed action prompts for motion and animation
            - Maintaining narrative coherence across complex multi-scene productions
            - Validating production feasibility and resource requirements""",
            
            verbose=True,
            allow_delegation=False,
            tools=[],
            
            # Agent-specific configuration
            config={
                "openai_api_key": self.config["OPENAI_API_KEY"],
                "model": "gpt-4o",
                "temperature": 0.3
            }
        )
    
    def create_voice_selection_agent(self) -> Agent:
        """Create the Voice Selection Agent"""
        
        return Agent(
            role="Voice Casting Director & ElevenLabs Specialist",
            goal="Analyze video content and select the most appropriate voice actor from ElevenLabs voice library based on content type, tone, and target audience",
            backstory="""You are an expert voice casting director with 20+ years of experience in the entertainment 
            industry, specializing in voice-over casting for commercials, documentaries, animations, and digital content. 
            You have deep knowledge of ElevenLabs voice library and understand how different voice characteristics 
            impact audience engagement and brand perception.
            
            You excel at:
            - Analyzing video content to determine optimal voice characteristics
            - Matching voice qualities to content type (advertisement, narrative, educational, etc.)
            - Understanding cultural and demographic considerations for voice selection
            - Navigating ElevenLabs voice library categories and filters
            - Recommending specific voice IDs based on content requirements
            - Considering factors like age, gender, accent, and emotional tone
            - Ensuring voice selection aligns with brand voice and target audience""",
            
            verbose=True,
            allow_delegation=False,
            tools=[],
            
            # Agent-specific configuration
            config={
                "openai_api_key": self.config["OPENAI_API_KEY"],
                "model": "gpt-4o",
                "temperature": 0.4
            }
        )
    
    def create_media_generation_agent(self) -> Agent:
        """Create the Media Generation Agent"""
        
        return Agent(
            role="AI Media Production Specialist",
            goal="Generate all media assets (images, videos, audio) using fal.ai APIs and orchestrate the complete media production pipeline",
            backstory="""You are a senior AI media production specialist with extensive experience in automated 
            content creation using cutting-edge AI platforms. You have worked with major content creation platforms, 
            advertising agencies, and media production houses. Your expertise spans image generation, video creation, 
            audio synthesis, and post-production workflows.
            
            You specialize in:
            - Generating high-quality images using fal.ai's Flux Pro models
            - Creating dynamic videos from static images with motion and animation
            - Producing natural-sounding voiceovers using ElevenLabs via fal.ai
            - Orchestrating complex media generation pipelines
            - Optimizing prompts for maximum AI generation quality
            - Managing concurrent generation tasks for efficiency
            - Ensuring technical quality and consistency across all media assets
            - Handling post-production tasks like video stitching and audio mixing""",
            
            verbose=True,
            allow_delegation=False,
            tools=[],
            
            # Agent-specific configuration
            config={
                "openai_api_key": self.config["OPENAI_API_KEY"],
                "fal_ai_api_key": self.config["FAL_AI_API_KEY"],
                "model": "gpt-4o",
                "temperature": 0.2
            }
        ) 