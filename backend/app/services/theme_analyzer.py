# backend/app/services/theme_analyzer.py
"""
ThemeAnalyzer: Thin wrapper around ClusterAnalysisService for theme extraction.
"""
from typing import List, Dict, Any
from backend.app.services.cluster_analysis_svc import ClusterAnalysisService

class ThemeAnalyzer:
    def __init__(self):
        self.cluster_service = ClusterAnalysisService()

    def identify_themes(self, collection_name: str) -> List[Dict[str, Any]]:
        """
        Returns a list of themes (cluster summaries) for a collection.
        Each theme: { 'theme_summary': str, 'supporting_reference_numbers': list }
        """
        analysis = self.cluster_service.analyze_collection(collection_name)
        cluster_summaries = analysis.get('cluster_summaries', {})
        # For each cluster, try to extract supporting doc reference numbers if possible
        themes = []
        for i, (cid, summary) in enumerate(cluster_summaries.items()):
            # Optionally, you could map cluster docs to ref nums if you want more granularity
            themes.append({
                'theme_summary': summary,
                'supporting_reference_numbers': []  # Could be filled with doc ref nums if needed
            })
        return themes

    def get_overall_summary(self, collection_name: str) -> str:
        analysis = self.cluster_service.analyze_collection(collection_name)
        return analysis.get('overall_summary', '')
