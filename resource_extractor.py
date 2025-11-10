"""
Azure resource ID parsing utilities.
Extracts subscription, resource group, type, and name from resource IDs.
"""

import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


class ResourceExtractor:
    """Parse Azure resource IDs into components."""
    
    @staticmethod
    def extract_resource_fields(resource_id: str) -> Tuple[str, str, str, str]:
        """
        Extract Azure resource components from resource ID.
        
        Args:
            resource_id: Full Azure resource ID
            
        Returns:
            Tuple of (subscription_id, resource_group, resource_type, resource_name)
            
        Example:
            resource_id = "/subscriptions/12345-abc/resourceGroups/prod-rg/providers/Microsoft.Storage/storageAccounts/prodstg001"
            -> ("12345-abc", "prod-rg", "Microsoft.Storage/storageAccounts", "prodstg001")
        """
        try:
            if not resource_id or not isinstance(resource_id, str):
                logger.warning("Empty or invalid resource_id provided")
                return "", "", "", ""
            
            # Extract subscription ID
            subscription_id = ""
            if '/subscriptions/' in resource_id:
                sub_start = resource_id.find('/subscriptions/') + 15
                sub_end = resource_id.find('/', sub_start)
                if sub_end == -1:
                    sub_end = len(resource_id)
                subscription_id = resource_id[sub_start:sub_end]
            
            # Extract resource group
            resource_group = ""
            if '/resourceGroups/' in resource_id:
                rg_start = resource_id.find('/resourceGroups/') + 16
                rg_end = resource_id.find('/', rg_start)
                if rg_end == -1:
                    rg_end = len(resource_id)
                resource_group = resource_id[rg_start:rg_end]
            
            # Extract resource type and name
            resource_type = ""
            resource_name = ""
            
            if '/providers/' in resource_id:
                prov_start = resource_id.find('/providers/') + 11
                
                # Split remaining path
                remaining = resource_id[prov_start:]
                parts = remaining.split('/')
                
                if len(parts) >= 3:
                    # Resource type is provider/resourceType (e.g., Microsoft.Storage/storageAccounts)
                    resource_type = f"{parts[0]}/{parts[1]}"
                    # Resource name is the next part
                    resource_name = parts[2]
                    
                    # Handle nested resources (e.g., servers/databases)
                    if len(parts) > 3:
                        # For nested resources like Microsoft.Sql/servers/databases
                        # Full path: Microsoft.Sql/servers/{serverName}/databases/{dbName}
                        nested_parts = parts[3:]
                        if nested_parts:
                            resource_name = '/'.join([parts[2]] + nested_parts)
            
            logger.debug(f"Extracted: sub={subscription_id}, rg={resource_group}, type={resource_type}, name={resource_name}")
            
            return subscription_id, resource_group, resource_type, resource_name
            
        except Exception as e:
            logger.error(f"Failed to extract resource fields from '{resource_id}': {str(e)}")
            return "", "", "", ""
    
    @staticmethod
    def extract_from_incident(incident_data: Dict[str, Any]) -> Tuple[str, str, str, str]:
        """
        Extract resource fields from incident data.
        
        Args:
            incident_data: Parsed incident JSON
            
        Returns:
            Tuple of (subscription_id, resource_group, resource_type, resource_name)
        """
        if not isinstance(incident_data, dict):
            logger.warning("incident_data is not a dictionary")
            return "", "", "", ""
        
        # Try to find resourceId in different locations
        resource_id = incident_data.get('resourceId', '')
        
        if not resource_id:
            # Try nested properties
            properties = incident_data.get('properties', {})
            if isinstance(properties, dict):
                resource_id = properties.get('resourceId', '')
        
        if not resource_id:
            # Try data field (some Azure logs have this structure)
            data = incident_data.get('data', {})
            if isinstance(data, dict):
                resource_id = data.get('resourceId', '')
        
        if resource_id:
            return ResourceExtractor.extract_resource_fields(resource_id)
        else:
            logger.debug("No resourceId found in incident data")
            return "", "", "", ""
    
    @staticmethod
    def get_resource_display_name(subscription_id: str, resource_group: str, 
                                  resource_type: str, resource_name: str) -> str:
        """
        Create a human-readable display name for the resource.
        
        Args:
            subscription_id: Azure subscription ID
            resource_group: Resource group name
            resource_type: Resource type (e.g., Microsoft.Storage/storageAccounts)
            resource_name: Resource name
            
        Returns:
            Formatted display name
        """
        if not resource_name:
            return "Unknown Resource"
        
        # Extract simple type name
        simple_type = resource_type.split('/')[-1] if resource_type else "Resource"
        
        return f"{simple_type}: {resource_name} (RG: {resource_group})"
    
    @staticmethod
    def is_production_resource(resource_id: str, resource_group: str = "", 
                              subscription_id: str = "") -> bool:
        """
        Determine if a resource is likely a production resource based on naming.
        
        Args:
            resource_id: Full resource ID
            resource_group: Resource group name
            subscription_id: Subscription ID
            
        Returns:
            True if resource appears to be production
        """
        resource_id_lower = resource_id.lower() if resource_id else ""
        rg_lower = resource_group.lower() if resource_group else ""
        
        # Check for production indicators
        prod_indicators = ['prod', 'production', 'prd']
        non_prod_indicators = ['dev','development' 'test', 'uat', 'staging', 'qa', 'sandbox']
        
        # Check resource ID
        for indicator in prod_indicators:
            if indicator in resource_id_lower or indicator in rg_lower:
                # Make sure it's not negated (like "non-prod")
                if 'non-' + indicator not in resource_id_lower:
                    return True
        
        # Check for non-production indicators
        for indicator in non_prod_indicators:
            if indicator in resource_id_lower or indicator in rg_lower:
                return False
        
        # Default to unknown (treat as production for safety)
        return False