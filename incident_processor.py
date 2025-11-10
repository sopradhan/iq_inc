"""
Main incident processing orchestrator.
Processes incidents through severity analysis and stores results.
"""

import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import joblib

from database_operations import DatabaseOperations
from bert_embedder import BertEmbedder
from severity_engine import SeverityEngine
from resource_extractor import ResourceExtractor

logger = logging.getLogger(__name__)


class IncidentProcessor:
    """Orchestrates incident processing pipeline."""
    
    def __init__(self, db_path: str, model_dir: str):
        """
        Initialize incident processor.
        
        Args:
            db_path: Path to SQLite database
            model_dir: Path to BERT model directory
        """
        self.db_path = db_path
        self.model_dir = model_dir
        
        # Initialize components
        logger.info("=" * 80)
        logger.info("Initializing Incident Processor Components")
        logger.info("=" * 80)
        
        try:
            logger.info("1. Initializing database operations...")
            self.db_ops = DatabaseOperations(db_path)
            logger.info("   ✓ Database operations initialized")
            
            logger.info("2. Initializing BERT embedder...")
            self.embedder = BertEmbedder(model_dir)
            logger.info("   ✓ BERT embedder initialized")
            
            logger.info("3. Initializing severity engine...")
            self.severity_engine = SeverityEngine(self.db_ops, self.embedder)
            logger.info("   ✓ Severity engine initialized")
            
            # Load ML models if available
            logger.info("4. Loading ML models...")
            self.scaler, self.classifier, self.weights = self._load_ml_models()
            
            logger.info("=" * 80)
            logger.info("Incident processor initialized successfully")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Failed to initialize incident processor: {str(e)}")
            raise
    
    def _load_ml_models(self) -> tuple:
        """
        Load trained ML models and weights.
        
        Returns:
            Tuple of (scaler, classifier, weights)
        """
        model_path = Path(self.model_dir)
        scaler_path = model_path / "incident_severity_scaler.joblib"
        classifier_path = model_path / "incident_severity_classifier.joblib"
        weights_path = model_path / "severity_weights.json"
        
        try:
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler not found: {scaler_path}")
            if not classifier_path.exists():
                raise FileNotFoundError(f"Classifier not found: {classifier_path}")
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights not found: {weights_path}")
            
            scaler = joblib.load(str(scaler_path))
            classifier = joblib.load(str(classifier_path))
            
            with open(str(weights_path), 'r') as f:
                weights = json.load(f)
            
            logger.info("   ✓ ML models loaded successfully")
            logger.info(f"     - Scaler: {scaler_path.name}")
            logger.info(f"     - Classifier: {classifier_path.name}")
            logger.info(f"     - Weights: {weights}")
            
            return scaler, classifier, weights
            
        except Exception as e:
            logger.warning(f"   ⚠ ML models not available: {str(e)}")
            logger.info("   → Using heuristic-based classification")
            
            # Default weights for heuristic classification
            weights = {
                'bert_weight': 0.35,
                'rule_weight': 0.45,
                'env_weight': 0.20,
                'intercept': 0.0
            }
            
            return None, None, weights
    
    def process_incidents(self, limit: Optional[int] = None, 
                         batch_size: int = 100) -> Dict[str, int]:
        """
        Process unprocessed incidents from database.
        
        Args:
            limit: Maximum number of incidents to process
            batch_size: Number of incidents to commit per batch
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info("=" * 80)
        logger.info("Starting Incident Processing")
        logger.info("=" * 80)
        
        # Fetch unprocessed incidents
        incidents = self.db_ops.get_unprocessed_incidents(limit=limit)
        
        if not incidents:
            logger.info("No unprocessed incidents found")
            return {"total": 0, "processed": 0, "failed": 0, "skipped": 0}
        
        stats = {
            "total": len(incidents),
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "by_severity": {"S1": 0, "S2": 0, "S3": 0, "S4": 0}
        }
        
        logger.info(f"Found {stats['total']} unprocessed incidents")
        logger.info("-" * 80)
        
        for idx, (incident_id, incident_json, source_type) in enumerate(incidents, 1):
            try:
                # Parse incident JSON
                try:
                    incident_data = json.loads(incident_json)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON for incident {incident_id}: {str(e)}")
                    stats['skipped'] += 1
                    continue
                
                # Analyze severity
                severity_result = self.severity_engine.analyze_incident(
                    incident_data,
                    scaler=self.scaler,
                    classifier=self.classifier,
                    weights=self.weights
                )
                
                # Extract resource information
                subscription_id, resource_group, resource_type, resource_name = \
                    ResourceExtractor.extract_from_incident(incident_data)
                
                # Prepare mapping data
                mapping_data = {
                    'payload_id': incident_id,
                    'subscription_id': subscription_id,
                    'resource_group': resource_group,
                    'resource_type': resource_type,
                    'resource_name': resource_name,
                    'environment': severity_result['environment'],
                    'severity_id': severity_result['severity_level'],
                    'bert_score': severity_result['bert_score'],
                    'rule_score': severity_result['rule_score'],
                    'combined_score': severity_result['combined_score'],
                    'matched_pattern': severity_result['matched_pattern'],
                    'is_incident': 1 if source_type == 'ActivityLog' else 0,
                    'source_type': source_type,
                    'payload': incident_json
                }
                
                # Insert severity mapping
                self.db_ops.insert_severity_mapping(mapping_data)
                
                # Update incident status
                self.db_ops.update_incident_status(incident_id, 'processed')
                
                # Update statistics
                stats['processed'] += 1
                severity_level = severity_result['severity_level']
                if severity_level in stats['by_severity']:
                    stats['by_severity'][severity_level] += 1
                
                # Log progress
                if idx % batch_size == 0:
                    logger.info(f"Progress: {idx}/{stats['total']} incidents processed")
                    logger.info(f"  Current stats: S1={stats['by_severity']['S1']}, "
                              f"S2={stats['by_severity']['S2']}, "
                              f"S3={stats['by_severity']['S3']}, "
                              f"S4={stats['by_severity']['S4']}")
                
            except Exception as e:
                logger.error(f"Failed to process incident {incident_id}: {str(e)}", exc_info=True)
                stats['failed'] += 1
                continue
        
        # Final summary
        logger.info("=" * 80)
        logger.info("Processing Complete")
        logger.info("=" * 80)
        logger.info(f"Total incidents: {stats['total']}")
        logger.info(f"Successfully processed: {stats['processed']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Skipped (invalid JSON): {stats['skipped']}")
        logger.info("-" * 80)
        logger.info("Severity Distribution:")
        for severity, count in sorted(stats['by_severity'].items()):
            percentage = (count / stats['processed'] * 100) if stats['processed'] > 0 else 0
            logger.info(f"  {severity}: {count} ({percentage:.1f}%)")
        logger.info("=" * 80)
        
        return stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with severity distribution and counts
        """
        severity_stats = self.db_ops.get_severity_statistics()
        
        total = sum(severity_stats.values())
        
        return {
            'severity_distribution': severity_stats,
            'total_processed': total,
            'percentages': {
                severity: (count / total * 100) if total > 0 else 0
                for severity, count in severity_stats.items()
            }
        }
    
    def reprocess_by_severity(self, severity_level: str, limit: int = 100) -> Dict[str, int]:
        """
        Reprocess incidents of a specific severity level.
        
        Args:
            severity_level: Severity level to reprocess (S1, S2, S3, S4)
            limit: Maximum number to reprocess
            
        Returns:
            Processing statistics
        """
        logger.info(f"Reprocessing {severity_level} incidents (limit: {limit})")
        
        incidents = self.db_ops.get_incidents_by_severity(severity_level, limit)
        
        stats = {"total": len(incidents), "reprocessed": 0, "failed": 0}
        
        for incident in incidents:
            try:
                incident_data = json.loads(incident['payload'])
                
                # Reanalyze
                severity_result = self.severity_engine.analyze_incident(
                    incident_data,
                    scaler=self.scaler,
                    classifier=self.classifier,
                    weights=self.weights
                )
                
                # Update if severity changed
                if severity_result['severity_level'] != incident['severity_id']:
                    logger.info(f"Severity changed for {incident['payload_id']}: "
                              f"{incident['severity_id']} -> {severity_result['severity_level']}")
                
                stats['reprocessed'] += 1
                
            except Exception as e:
                logger.error(f"Failed to reprocess {incident['payload_id']}: {str(e)}")
                stats['failed'] += 1
        
        return stats