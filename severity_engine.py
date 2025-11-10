"""
Severity classification engine with rule-based and ML-based analysis.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


class SeverityEngine:
    """Incident severity classification engine."""
    
    def __init__(self, db_ops, embedder):
        """
        Initialize severity engine.
        
        Args:
            db_ops: DatabaseOperations instance
            embedder: BertEmbedder instance
        """
        self.db_ops = db_ops
        self.embedder = embedder
        self.similarity_threshold = 0.65
        self.pattern_info = {}
        
        self._load_and_encode_rules()
    
    def _load_and_encode_rules(self) -> None:
        """Load severity rules and encode patterns."""
        logger.info("Loading and encoding severity rules...")
        
        rules = self.db_ops.load_severity_rules()
        
        encoded_count = 0
        for pattern, severity_level, base_score, category, description, environment in rules:
            try:
                # Encode pattern
                pattern_embedding = self.embedder.encode(pattern.lower())
                
                self.pattern_info[pattern] = {
                    'embedding': pattern_embedding,
                    'severity_level': severity_level,
                    'base_score': base_score,
                    'category': category,
                    'description': description,
                    'environment': environment
                }
                
                encoded_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to encode pattern '{pattern}': {str(e)}")
                continue
        
        logger.info(f"Successfully encoded {encoded_count}/{len(rules)} severity patterns")
        
        if encoded_count == 0:
            raise RuntimeError("No severity patterns could be encoded. Check BERT embedder and database.")
    
    def extract_analysis_text(self, incident_data: Dict[str, Any]) -> str:
        """
        Extract relevant text from incident for analysis.
        
        Args:
            incident_data: Parsed incident JSON
            
        Returns:
            Combined analysis text
        """
        text_parts = []
        
        if not isinstance(incident_data, dict):
            logger.warning("incident_data is not a dictionary")
            return ""
        
        # Extract error message
        properties = incident_data.get("properties", {})
        if isinstance(properties, dict):
            error = properties.get("error", {})
            if isinstance(error, dict):
                error_msg = error.get("message", "")
                error_code = error.get("code", "")
                if error_msg:
                    text_parts.append(f"error: {error_msg}")
                if error_code:
                    text_parts.append(f"code: {error_code}")
        
        # Extract operation name
        operation = incident_data.get("operationName")
        if operation:
            if isinstance(operation, dict):
                op_value = operation.get("localizedValue") or operation.get("value", "")
                if op_value:
                    text_parts.append(f"operation: {op_value}")
            else:
                text_parts.append(f"operation: {operation}")
        
        # Extract status
        status = incident_data.get("status")
        if status:
            if isinstance(status, dict):
                status_value = status.get("localizedValue") or status.get("value", "")
                if status_value:
                    text_parts.append(f"status: {status_value}")
            else:
                text_parts.append(f"status: {status}")
        
        # Extract category
        category = incident_data.get("category")
        if category:
            text_parts.append(f"category: {category}")
        
        # Extract metric name for metric logs
        metric_name = incident_data.get("metricName", "")
        if metric_name:
            text_parts.append(f"metric: {metric_name}")
        
        # Extract service/tags
        tags = incident_data.get("tags", {})
        if isinstance(tags, dict):
            service = tags.get("Service", "")
            if service:
                text_parts.append(f"service: {service}")
        
        combined_text = " ".join(filter(None, text_parts))
        
        if not combined_text:
            # Fallback to ID or first 100 chars of JSON
            combined_text = str(incident_data.get("id", ""))[:100]
            if not combined_text:
                combined_text = str(incident_data)[:100]
        
        return combined_text
    
    def get_environment_factor(self, incident_data: Dict[str, Any]) -> Tuple[float, str]:
        """
        Determine environment factor from incident data.
        
        Args:
            incident_data: Parsed incident JSON
            
        Returns:
            Tuple of (environment_factor, environment_name)
        """
        environment = None
        
        if not isinstance(incident_data, dict):
            return 1.0, "unknown"
        
        # Check properties.environment
        properties = incident_data.get("properties", {})
        if isinstance(properties, dict):
            env_value = properties.get("environment") or properties.get("env")
            if env_value:
                environment = str(env_value).lower()
        
        # Check tags.Environment
        if not environment:
            tags = incident_data.get("tags", {})
            if isinstance(tags, dict):
                env_tag = tags.get("Environment", "")
                if env_tag:
                    environment = str(env_tag).lower()
        
        # Check resource ID patterns
        if not environment:
            resource_id = str(incident_data.get("resourceId", "")).lower()
            if "prod-" in resource_id or "/prod/" in resource_id:
                environment = "prod"
            elif "uat-" in resource_id or "/uat/" in resource_id:
                environment = "uat"
            elif "dev-" in resource_id or "/dev/" in resource_id:
                environment = "dev"
        
        # Map to factor
        env_factor_map = {
            "prod": (1.2, "prod"),
            "production": (1.2, "prod"),
            "uat": (1.0, "uat"),
            "staging": (1.0, "uat"),
            "test": (1.0, "uat"),
            "dev": (0.8, "dev"),
            "development": (0.8, "dev")
        }
        
        return env_factor_map.get(environment, (1.0, "unknown"))
    
    def analyze_incident(self, incident_data: Dict[str, Any], 
                        scaler=None, classifier=None, 
                        weights: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze incident and determine severity.
        
        Args:
            incident_data: Parsed incident JSON
            scaler: Trained feature scaler (optional)
            classifier: Trained classifier model (optional)
            weights: Feature weights dictionary (optional)
            
        Returns:
            Dictionary containing severity analysis results
        """
        # Extract and encode text
        analysis_text = self.extract_analysis_text(incident_data)
        
        if not analysis_text or len(analysis_text.strip()) < 3:
            logger.warning(f"Empty or too short analysis text: '{analysis_text}', defaulting to S4")
            return self._default_severity_result()
        
        try:
            text_embedding = self.embedder.encode(analysis_text.lower())
        except Exception as e:
            logger.error(f"Failed to encode text '{analysis_text[:50]}...': {str(e)}")
            return self._default_severity_result()
        
        # Find matching patterns
        matches = []
        for pattern, info in self.pattern_info.items():
            try:
                similarity = 1.0 - cosine(text_embedding, info['embedding'])
                
                if similarity >= self.similarity_threshold:
                    matches.append({
                        'pattern': pattern,
                        'similarity': similarity,
                        'severity_level': info['severity_level'],
                        'base_score': info['base_score'],
                        'category': info['category']
                    })
            except Exception as e:
                logger.warning(f"Failed to compute similarity for pattern '{pattern}': {str(e)}")
                continue
        
        if not matches:
            logger.info(f"No pattern matches found for text: '{analysis_text[:100]}...', defaulting to S4")
            return self._default_severity_result()
        
        # Sort by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        best_match = matches[0]
        
        logger.debug(f"Best match: pattern='{best_match['pattern']}', similarity={best_match['similarity']:.4f}")
        
        # Get environment factor
        env_factor, env_name = self.get_environment_factor(incident_data)
        
        # Calculate scores
        bert_score = best_match['similarity']
        rule_score = float(best_match['base_score'])
        adjusted_rule_score = rule_score * env_factor
        
        # Extract additional features for ML model (if available)
        properties = incident_data.get("properties", {})
        error_rate = properties.get("error_rate", 0.0) if isinstance(properties, dict) else 0.0
        latency = properties.get("latency_ms", 0.0) if isinstance(properties, dict) else 0.0
        availability = properties.get("availability_percent", 100.0) if isinstance(properties, dict) else 100.0
        
        # Normalize additional features
        error_rate_norm = min(error_rate / 100.0, 1.0)
        latency_norm = min(latency / 10000.0, 1.0)
        availability_norm = 1.0 - (availability / 100.0)
        
        # Determine severity
        if classifier is not None and scaler is not None and weights is not None:
            # ML-based classification with 6 features
            severity_level, combined_score = self._ml_classify(
                bert_score, adjusted_rule_score, env_factor,
                error_rate_norm, latency_norm, availability_norm,
                scaler, classifier, weights
            )
        else:
            # Heuristic classification
            severity_level, combined_score = self._heuristic_classify(
                bert_score, adjusted_rule_score, weights
            )
        
        logger.info(f"Classified as {severity_level}: bert={bert_score:.4f}, rule={adjusted_rule_score:.2f}, combined={combined_score:.2f}")
        
        return {
            "severity_level": severity_level,
            "bert_score": float(bert_score),
            "rule_score": float(adjusted_rule_score),
            "combined_score": float(combined_score),
            "matched_pattern": best_match['pattern'],
            "environment": env_name,
            "all_matches": matches[:3]
        }
    
    def _ml_classify(self, bert_score: float, rule_score: float, env_factor: float,
                     error_rate: float, latency: float, availability: float,
                     scaler, classifier, weights: Dict) -> Tuple[str, float]:
        """ML-based severity classification with 6 features."""
        try:
            # Create feature vector with 6 features (matching training)
            features = np.array([[
                bert_score,           # Feature 1: BERT similarity
                rule_score,           # Feature 2: Rule base score
                env_factor,           # Feature 3: Environment factor
                error_rate,           # Feature 4: Error rate (normalized)
                latency,              # Feature 5: Latency (normalized)
                availability          # Feature 6: Availability impact (normalized)
            ]])
            
            features_scaled = scaler.transform(features)
            severity_class = classifier.predict(features_scaled)[0]
            
            combined_score = (
                weights['bert_weight'] * bert_score * 100 +
                weights['rule_weight'] * rule_score +
                weights['env_weight'] * env_factor * 10 +
                weights.get('intercept', 0)
            )
            
            severity_map = {0: 'S1', 1: 'S2', 2: 'S3', 3: 'S4'}
            severity_level = severity_map.get(severity_class, 'S4')
            
            logger.debug(f"ML classification: class={severity_class}, severity={severity_level}")
            
            return severity_level, combined_score
            
        except Exception as e:
            logger.error(f"ML classification failed: {str(e)}, falling back to heuristic")
            return self._heuristic_classify(bert_score, rule_score, weights)
    
    def _heuristic_classify(self, bert_score: float, rule_score: float,
                           weights: Optional[Dict]) -> Tuple[str, float]:
        """Heuristic severity classification."""
        bert_weight = weights.get('bert_weight', 0.4) if weights else 0.4
        rule_weight = weights.get('rule_weight', 0.6) if weights else 0.6
        
        combined_score = (bert_score * 100 * bert_weight) + (rule_score * rule_weight)
        
        if combined_score >= 80:
            severity_level = "S1"
        elif combined_score >= 60:
            severity_level = "S2"
        elif combined_score >= 40:
            severity_level = "S3"
        else:
            severity_level = "S4"
        
        logger.debug(f"Heuristic classification: score={combined_score:.2f}, severity={severity_level}")
        
        return severity_level, combined_score
    
    def _default_severity_result(self) -> Dict[str, Any]:
        """Return default severity result for S4."""
        return {
            "severity_level": "S4",
            "bert_score": 0.0,
            "rule_score": 10.0,
            "combined_score": 10.0,
            "matched_pattern": None,
            "environment": "unknown",
            "all_matches": []
        }