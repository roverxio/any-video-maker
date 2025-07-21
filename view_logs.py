#!/usr/bin/env python3
"""
Log Viewer Utility
Helps you easily view logs from video generation runs
"""

import argparse
import os
from pathlib import Path
from datetime import datetime


def list_recent_runs(outputs_dir: Path, limit: int = 10):
    """List recent video generation runs"""
    
    if not outputs_dir.exists():
        print("❌ No outputs directory found. Run a video generation first.")
        return []
    
    # Get all run folders
    run_folders = [
        folder for folder in outputs_dir.iterdir() 
        if folder.is_dir() and folder.name.startswith('video_gen_')
    ]
    
    # Sort by modification time (newest first)
    run_folders.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not run_folders:
        print("❌ No video generation runs found.")
        return []
    
    print(f"📁 Found {len(run_folders)} recent runs:\n")
    
    for i, folder in enumerate(run_folders[:limit], 1):
        # Get modification time
        mod_time = datetime.fromtimestamp(folder.stat().st_mtime)
        
        # Check for key files
        logs_exist = {
            'generation.log': (folder / 'generation.log').exists(),
            'summary.log': (folder / 'summary.log').exists(),
            'errors.log': (folder / 'errors.log').exists()
        }
        
        # Check for final video
        has_video = any((folder / 'media').glob('*.mp4')) if (folder / 'media').exists() else False
        
        status = "✅ Complete" if has_video else "⏳ Partial"
        log_status = " | ".join([f"{log}: {'✓' if exists else '✗'}" for log, exists in logs_exist.items()])
        
        print(f"{i:2d}. {folder.name}")
        print(f"    📅 {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"    📊 {status}")
        print(f"    📄 Logs: {log_status}")
        print()
    
    return run_folders[:limit]


def view_log_file(log_file: Path, tail_lines: int = None):
    """View a log file with optional tail functionality"""
    
    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        return
    
    print(f"📄 Viewing: {log_file}")
    print("=" * 80)
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if tail_lines and len(lines) > tail_lines:
            print(f"... (showing last {tail_lines} lines of {len(lines)} total) ...")
            lines = lines[-tail_lines:]
        
        for line in lines:
            print(line.rstrip())
            
    except Exception as e:
        print(f"❌ Error reading log file: {e}")


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="View logs from video generation runs")
    parser.add_argument("--run", "-r", help="Run folder name or number from list")
    parser.add_argument("--log", "-l", choices=['generation', 'summary', 'errors'], 
                       default='summary', help="Which log to view (default: summary)")
    parser.add_argument("--tail", "-t", type=int, help="Show only last N lines")
    parser.add_argument("--list", "-ls", action="store_true", help="List recent runs")
    parser.add_argument("--outputs-dir", default="outputs", help="Outputs directory (default: outputs)")
    
    args = parser.parse_args()
    
    outputs_dir = Path(args.outputs_dir)
    
    # List recent runs
    if args.list or not args.run:
        recent_runs = list_recent_runs(outputs_dir)
        if not args.run and recent_runs:
            print("💡 Use --run <number> to view logs from a specific run")
            print("💡 Example: python view_logs.py --run 1 --log summary")
        return
    
    # Find the specified run
    run_folder = None
    
    if args.run.isdigit():
        # Run number from list
        recent_runs = list_recent_runs(outputs_dir, limit=50)
        run_num = int(args.run)
        if 1 <= run_num <= len(recent_runs):
            run_folder = recent_runs[run_num - 1]
        else:
            print(f"❌ Invalid run number. Use 1-{len(recent_runs)}")
            return
    else:
        # Direct folder name
        run_folder = outputs_dir / args.run
        if not run_folder.exists():
            print(f"❌ Run folder not found: {run_folder}")
            return
    
    # Determine log file
    log_files = {
        'generation': run_folder / 'generation.log',
        'summary': run_folder / 'summary.log', 
        'errors': run_folder / 'errors.log'
    }
    
    log_file = log_files[args.log]
    
    print(f"📁 Run: {run_folder.name}")
    view_log_file(log_file, args.tail)


if __name__ == "__main__":
    main() 