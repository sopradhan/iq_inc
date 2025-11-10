"""
Main entry point for incident severity classification system.
Orchestrates all components and processes incidents.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

from incident_processor import IncidentProcessor


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure application-wide logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_dir / "classifier.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')


def validate_configuration(model_dir: str, db_path: str) -> None:
    """
    Validate configuration paths exist.
    
    Args:
        model_dir: Path to model directory
        db_path: Path to database file
        
    Raises:
        FileNotFoundError: If required files don't exist
    """
    logger = logging.getLogger(__name__)
    
    # Check database
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    logger.info(f"✓ Database found: {db_path}")
    
    # Check model directory
    if not Path(model_dir).exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    logger.info(f"✓ Model directory found: {model_dir}")
    
    # Check BERT model files
    required_bert_files = ['config.json', 'vocab.txt']
    for file in required_bert_files:
        file_path = Path(model_dir) / file
        if not file_path.exists():
            raise FileNotFoundError(f"Required BERT file not found: {file_path}")
    logger.info(f"✓ BERT model files validated")
    
    # Check for model weights (either format)
    has_weights = (
        (Path(model_dir) / 'pytorch_model.bin').exists() or
        (Path(model_dir) / 'model.safetensors').exists()
    )
    if not has_weights:
        raise FileNotFoundError(f"BERT model weights not found in {model_dir}")
    logger.info(f"✓ BERT model weights found")


def print_banner():
    """Print application banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║         Incident Severity Classification System              ║
    ║                                                               ║
    ║         Version: 2.0                                          ║
    ║         Using: BERT Embeddings + ML Classification           ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Main execution function."""
    
    # Print banner
    print_banner()
    
    # Setup logging
    setup_logging(log_level="INFO")
    logger = logging.getLogger(__name__)
    
    # Configuration
    MODEL_DIR = r"D:\incident_management\models\bert"
    DB_PATH = r"D:\incident_management\data\sqlite\incident_management_v2.db"
    
    # Processing options
    BATCH_SIZE = 100  # Commit every 100 incidents
    LIMIT = None  # Process all (set to number to limit)
    
    try:
        # Validate configuration
        logger.info("Validating configuration...")
        validate_configuration(MODEL_DIR, DB_PATH)
        logger.info("Configuration validated successfully")
        print()
        
        # Initialize processor
        logger.info("Initializing incident processor...")
        processor = IncidentProcessor(db_path=DB_PATH, model_dir=MODEL_DIR)
        print()
        
        # Process incidents
        logger.info("Starting incident processing...")
        stats = processor.process_incidents(limit=LIMIT, batch_size=BATCH_SIZE)
        print()
        
        # Display final results
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"Total incidents found:      {stats['total']}")
        print(f"Successfully processed:     {stats['processed']}")
        print(f"Failed:                     {stats['failed']}")
        print(f"Skipped (invalid JSON):     {stats['skipped']}")
        print("-" * 80)
        print("Severity Distribution:")
        
        if stats['processed'] > 0:
            for severity in ['S1', 'S2', 'S3', 'S4']:
                count = stats['by_severity'].get(severity, 0)
                percentage = (count / stats['processed'] * 100)
                bar_length = int(percentage / 2)  # Scale for 50 char width
                bar = '█' * bar_length
                print(f"  {severity}: {count:4d} ({percentage:5.1f}%) {bar}")
        else:
            print("  No incidents processed")
        
        print("=" * 80)
        
        # Get detailed statistics
        logger.info("\nFetching detailed statistics...")
        statistics = processor.get_statistics()
        
        print("\nDetailed Statistics:")
        print(f"Total processed in database: {statistics['total_processed']}")
        print("Severity breakdown:")
        for severity, count in sorted(statistics['severity_distribution'].items()):
            pct = statistics['percentages'].get(severity, 0)
            print(f"  {severity}: {count} ({pct:.1f}%)")
        
        print("\n" + "=" * 80)
        print("Classification complete!")
        print("=" * 80 + "\n")
        
        # Log completion
        logger.info("Incident classification completed successfully")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"Configuration error: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        print("Please check your configuration paths.")
        return 1
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(f"\n❌ Unexpected error occurred: {str(e)}")
        print("Check logs/classifier.log for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())