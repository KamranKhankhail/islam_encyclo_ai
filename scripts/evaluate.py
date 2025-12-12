"""
Evaluation Framework for Search Engine
Measures precision, recall, and accuracy
"""

import json
from pathlib import Path
from typing import List, Dict
from search_engine import QuranSearchEngine


class SearchEvaluator:
    """Evaluate search engine performance"""
    
    def __init__(self, engine: QuranSearchEngine, test_queries_path: str):
        """
        Initialize evaluator
        
        Args:
            engine: Initialized QuranSearchEngine
            test_queries_path: Path to test_queries.json
        """
        self.engine = engine
        
        # Load test queries
        with open(test_queries_path, 'r', encoding='utf-8') as f:
            self.test_queries = json.load(f)
        
        print(f"Loaded {len(self.test_queries)} test queries")
    
    def evaluate_query(self, query_obj: Dict, k: int = 5) -> Dict:
        """
        Evaluate a single query
        
        Args:
            query_obj: Query object with 'query' and 'expected' fields
            k: Number of results to retrieve
        
        Returns:
            Evaluation metrics for this query
        """
        query = query_obj['query']
        expected_verse = query_obj['expected']
        query_type = query_obj.get('type', 'unknown')
        
        # Perform search
        results = self.engine.search(query, top_k=k)
        
        # Check if expected verse is in results
        found_at_position = None
        for i, result in enumerate(results):
            if result['verse_key'] == expected_verse:
                found_at_position = i + 1  # 1-indexed
                break
        
        # Calculate metrics
        metrics = {
            'query': query,
            'expected': expected_verse,
            'type': query_type,
            'found': found_at_position is not None,
            'position': found_at_position,
            'precision_at_1': 1.0 if found_at_position == 1 else 0.0,
            'recall_at_k': 1.0 if found_at_position is not None else 0.0,
            'mrr': 1.0 / found_at_position if found_at_position else 0.0,  # Mean Reciprocal Rank
        }
        
        return metrics
    
    def evaluate_all(self, k: int = 5) -> Dict:
        """
        Evaluate all test queries
        
        Args:
            k: Number of results to retrieve per query
        
        Returns:
            Aggregated metrics
        """
        print(f"\nEvaluating {len(self.test_queries)} queries...")
        print("="*60)
        
        all_metrics = []
        
        for query_obj in self.test_queries:
            metrics = self.evaluate_query(query_obj, k)
            all_metrics.append(metrics)
            
            # Print progress
            status = "✓" if metrics['found'] else "✗"
            print(f"{status} {query_obj['query'][:40]:<40} | Found: {metrics['found']} @ pos {metrics['position']}")
        
        # Calculate aggregate metrics
        total = len(all_metrics)
        found_count = sum(1 for m in all_metrics if m['found'])
        
        aggregate = {
            'total_queries': total,
            'found_count': found_count,
            'accuracy': found_count / total if total > 0 else 0.0,
            'precision_at_1': sum(m['precision_at_1'] for m in all_metrics) / total,
            'recall_at_k': sum(m['recall_at_k'] for m in all_metrics) / total,
            'mrr': sum(m['mrr'] for m in all_metrics) / total,  # Mean Reciprocal Rank
        }
        
        # Break down by query type
        query_types = set(m['type'] for m in all_metrics)
        aggregate['by_type'] = {}
        
        for qtype in query_types:
            type_metrics = [m for m in all_metrics if m['type'] == qtype]
            if type_metrics:
                type_total = len(type_metrics)
                type_found = sum(1 for m in type_metrics if m['found'])
                aggregate['by_type'][qtype] = {
                    'total': type_total,
                    'found': type_found,
                    'accuracy': type_found / type_total if type_total > 0 else 0.0
                }
        
        return aggregate, all_metrics
    
    def print_report(self, aggregate: Dict, detailed: bool = False):
        """Print evaluation report"""
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        
        print(f"\nOverall Metrics:")
        print(f"  Total Queries: {aggregate['total_queries']}")
        print(f"  Found: {aggregate['found_count']} / {aggregate['total_queries']}")
        print(f"  Accuracy: {aggregate['accuracy']*100:.1f}%")
        print(f"  Precision@1: {aggregate['precision_at_1']*100:.1f}%")
        print(f"  Recall@K: {aggregate['recall_at_k']*100:.1f}%")
        print(f"  MRR: {aggregate['mrr']:.3f}")
        
        if aggregate.get('by_type'):
            print(f"\nAccuracy by Query Type:")
            for qtype, metrics in aggregate['by_type'].items():
                acc = metrics['accuracy'] * 100
                print(f"  {qtype:20s}: {acc:5.1f}% ({metrics['found']}/{metrics['total']})")
        
        print("\n" + "="*60)
        
        # Assessment
        accuracy = aggregate['accuracy'] * 100
        if accuracy >= 85:
            print("✓ EXCELLENT: Accuracy meets target (≥85%)")
        elif accuracy >= 70:
            print("⚠ GOOD: Accuracy acceptable for MVP (≥70%)")
        else:
            print("✗ NEEDS IMPROVEMENT: Accuracy below target (<70%)")
        
        print("="*60)


def run_evaluation():
    """Run complete evaluation"""
    # Paths
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "output" / "processed" / "quran_complete.json"
    test_queries_path = base_dir / "output" / "processed" / "test_queries.json"
    
    # Initialize engine
    print("Initializing search engine...")
    engine = QuranSearchEngine(str(data_path))
    
    # Initialize evaluator
    evaluator = SearchEvaluator(engine, str(test_queries_path))
    
    # Run evaluation
    aggregate, detailed = evaluator.evaluate_all(k=5)
    
    # Print report
    evaluator.print_report(aggregate)
    
    # Save results
    output_path = base_dir / "output" / "processed" / "evaluation_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'aggregate': aggregate,
            'detailed': detailed
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")
    
    return aggregate


if __name__ == "__main__":
    run_evaluation()