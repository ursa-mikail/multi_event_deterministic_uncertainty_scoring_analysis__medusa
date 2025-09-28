import yaml
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import math

@dataclass
class PremiseScore:
    name: str
    score: float
    description: str
    deviation: str

class ClaimsForecastScorer:
    def __init__(self):
        # Enhanced similarity matrix with better categorization
        self.similarity_matrix = {
            'vehicle': {
                'car': 1.0, 'truck': 0.8, 'motorcycle': 0.7, 'bicycle': 0.6, 'bus': 0.8,
                'toy_car': 0.1, 'cloud': 0.0, 'animal': 0.0, 'camry': 1.0, 'toyota': 1.0
            },
            'car': {
                'car': 1.0, 'vehicle': 1.0, 'automobile': 1.0, 'sedan': 0.9, 'truck': 0.6,
                'motorcycle': 0.4, 'camry': 1.0, 'toyota': 1.0, 'red': 1.0, 'blue': 1.0,
                'toy_car': 0.1, 'real_car': 1.0
            },
            'monument': {
                'eiffel_tower': 1.0, 'louvre': 0.9, 'notre_dame': 0.9, 'arc_de_triomphe': 0.9,
                'building': 0.8, 'statue': 0.8, 'landmark': 1.0, 'tower': 1.0, 'monument': 1.0
            },
            'landmark': {
                'landmark': 1.0, 'monument': 1.0, 'eiffel_tower': 1.0, 'louvre': 1.0,
                'notre_dame': 1.0, 'building': 0.9, 'tower': 1.0, 'statue': 0.9
            }
        }
        
        self.action_similarity = {
            'collide': {'collide': 1.0, 'crash': 0.95, 'bump': 0.8, 'hit': 0.9, 'accident': 0.85, 'incident': 0.8, 'park': 0.1},
            'crash': {'crash': 1.0, 'collide': 0.95, 'bump': 0.7, 'hit': 0.9, 'accident': 0.9},
            'bump': {'bump': 1.0, 'collide': 0.8, 'hit': 0.85, 'crash': 0.7, 'accident': 0.7},
            'incident': {'incident': 1.0, 'accident': 0.9, 'crash': 0.8, 'collide': 0.8, 'event': 0.9},
            'accident': {'accident': 1.0, 'incident': 0.9, 'crash': 0.9, 'collide': 0.85}
        }
        
        # Define category hierarchies
        self.category_hierarchies = {
            'vehicle': ['car', 'truck', 'motorcycle', 'bus', 'bicycle', 'vehicle', 'automobile'],
            'monument': ['eiffel_tower', 'louvre', 'notre_dame', 'arc_de_triomphe', 'monument', 'landmark', 'tower', 'building'],
            'landmark': ['landmark', 'monument', 'eiffel_tower', 'louvre', 'notre_dame', 'tower', 'building'],
            'car': ['car', 'vehicle', 'automobile', 'sedan', 'camry', 'toyota', 'honda', 'ford']
        }
    
    def calculate_time_score(self, predicted: str, actual: str) -> PremiseScore:
        """Calculate time accuracy score"""
        try:
            # Exact string match
            if predicted.lower() == actual.lower():
                return PremiseScore("time", 1.0, "Exact time match", "No deviation")
            
            pred_date = self._parse_date(predicted)
            actual_date = self._parse_date(actual)
            
            if pred_date == actual_date:
                return PremiseScore("time", 1.0, "Exact date match", "No deviation")
            
            days_diff = abs((actual_date - pred_date).days)
            
            # Different scoring strategies based on prediction specificity
            if "on" in predicted.lower() or re.search(r'\d{1,2}/\d{1,2}/\d{4}', predicted):
                # Specific date prediction
                score = max(0, 1 - (days_diff / 30))  # 30-day window
                deviation = f"Off by {days_diff} days from specific date"
                return PremiseScore("time", round(score, 2), "Specific date accuracy", deviation)
            
            elif "in" in predicted.lower() and any(month in predicted.lower() for month in 
                 ['january', 'february', 'march', 'april', 'may', 'june', 'july', 
                  'august', 'september', 'october', 'november', 'december']):
                # Month prediction
                same_month = pred_date.month == actual_date.month
                same_year = pred_date.year == actual_date.year
                score = 1.0 if same_month and same_year else 0.3
                deviation = "Month accuracy" if score == 1.0 else "Wrong month"
                return PremiseScore("time", score, "Month accuracy", deviation)
            
            elif "in" in predicted.lower() and any(year in predicted for year in ['2023', '2024', '2025']):
                # Year prediction
                same_year = pred_date.year == actual_date.year
                score = 1.0 if same_year else 0.1
                deviation = "Year accuracy" if score == 1.0 else "Wrong year"
                return PremiseScore("time", score, "Year accuracy", deviation)
            
            else:
                # Generic time reference - be more lenient
                score = max(0.5, 1 - (days_diff / 365))  # Minimum 0.5 for generic references
                deviation = f"Generic time reference off by {days_diff} days"
                return PremiseScore("time", round(score, 2), "Generic time accuracy", deviation)
            
        except Exception as e:
            return PremiseScore("time", 0.5, "Time accuracy", f"Generic time match - cannot parse specific dates")
    
    def calculate_location_score(self, predicted: str, actual: str) -> PremiseScore:
        """Calculate location accuracy score"""
        pred_loc = predicted.lower()
        actual_loc = actual.lower()
        
        # Exact match
        if pred_loc == actual_loc:
            return PremiseScore("location", 1.0, "Exact location match", "No deviation")
        
        # Contains match (e.g., "Paris" in "Eiffel Tower, Paris")
        if pred_loc in actual_loc or actual_loc in pred_loc:
            return PremiseScore("location", 1.0, "Location contains match", "No deviation")
        
        # Location hierarchy and proximity scoring
        location_hierarchy = {
            'eiffel_tower': ['paris', 'france', 'europe', 'monument', 'landmark', 'tower'],
            'louvre': ['paris', 'france', 'europe', 'monument', 'landmark', 'museum'],
            'notre_dame': ['paris', 'france', 'europe', 'monument', 'landmark', 'cathedral'],
            'paris': ['france', 'europe', 'city'],
            'france': ['europe', 'country'],
            'monument': ['landmark', 'eiffel_tower', 'louvre', 'notre_dame'],
            'landmark': ['monument', 'eiffel_tower', 'louvre', 'notre_dame']
        }
        
        # Check if locations are related in hierarchy
        for location, parents in location_hierarchy.items():
            if location in pred_loc:
                # Check if actual location is this location or any of its parents/related
                if any(parent in actual_loc for parent in [location] + parents):
                    return PremiseScore("location", 1.0, "Location hierarchy match", "No deviation")
            
            if location in actual_loc:
                # Check if predicted location is this location or any of its parents/related
                if any(parent in pred_loc for parent in [location] + parents):
                    return PremiseScore("location", 1.0, "Location hierarchy match", "No deviation")
        
        # Same city but different specific location
        if 'paris' in pred_loc and 'paris' in actual_loc:
            return PremiseScore("location", 0.9, "Same city, different location", "Different specific location in same city")
        
        # Same country but different city
        if 'france' in pred_loc and 'france' in actual_loc:
            return PremiseScore("location", 0.8, "Same country, different city", "Different city in same country")
        
        # Same continent
        if any(continent in pred_loc and continent in actual_loc for continent in ['europe', 'asia', 'america']):
            return PremiseScore("location", 0.6, "Same continent", "Different country but same continent")
        
        return PremiseScore("location", 0.3, "Different location", "Completely different location")
    
    def calculate_entity_score(self, predicted: str, actual: str, entity_type: str = "subject") -> PremiseScore:
        """Calculate entity (subject/object) accuracy score"""
        pred_clean = predicted.lower().strip()
        actual_clean = actual.lower().strip()
        
        # Exact match
        if pred_clean == actual_clean:
            return PremiseScore(entity_type, 1.0, f"Exact {entity_type} match", "No deviation")
        
        # Contains match (e.g., "car" in "red Toyota Camry")
        if pred_clean in actual_clean or actual_clean in pred_clean:
            return PremiseScore(entity_type, 1.0, f"{entity_type} contains match", "No deviation")
        
        # Check similarity matrix with individual word matching
        pred_words = re.findall(r'\w+', pred_clean)
        actual_words = re.findall(r'\w+', actual_clean)
        
        # Check each word in prediction against each word in actual
        best_similarity = 0.0
        best_match_description = ""
        
        for p_word in pred_words:
            for a_word in actual_words:
                for category, similarities in self.similarity_matrix.items():
                    if p_word in similarities and a_word in similarities:
                        similarity = similarities[a_word]
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_description = f"Word similarity: '{p_word}' vs '{a_word}' in category '{category}'"
        
        if best_similarity > 0.8:
            return PremiseScore(entity_type, best_similarity, f"High conceptual similarity", best_match_description)
        elif best_similarity > 0.5:
            return PremiseScore(entity_type, best_similarity, f"Moderate conceptual similarity", best_match_description)
        
        # Category hierarchy matching
        for category, items in self.category_hierarchies.items():
            pred_in_category = any(item in pred_clean for item in items)
            actual_in_category = any(item in actual_clean for item in items)
            
            if pred_in_category and actual_in_category:
                # Both in same broad category
                if category in ['vehicle', 'car'] and entity_type == 'subject':
                    return PremiseScore(entity_type, 1.0, f"Same {category} category", f"Both are {category}s")
                elif category in ['monument', 'landmark'] and entity_type == 'object':
                    return PremiseScore(entity_type, 1.0, f"Same {category} category", f"Both are {category}s")
        
        # Final fallback - check if they share any common words
        common_words = set(pred_words) & set(actual_words)
        if common_words:
            return PremiseScore(entity_type, 0.7, "Shared descriptive words", f"Common words: {', '.join(common_words)}")
        
        return PremiseScore(entity_type, 0.5, "Different specific entity", f"Predicted '{pred_clean}' vs actual '{actual_clean}'")
    
    def calculate_action_score(self, predicted: str, actual: str) -> PremiseScore:
        """Calculate action/event accuracy score"""
        pred_clean = predicted.lower().strip()
        actual_clean = actual.lower().strip()
        
        # Exact match
        if pred_clean == actual_clean:
            return PremiseScore("action", 1.0, "Exact action match", "No deviation")
        
        # Contains match
        if pred_clean in actual_clean or actual_clean in pred_clean:
            return PremiseScore("action", 1.0, "Action contains match", "No deviation")
        
        # Check action similarity matrix with word-level matching
        pred_words = re.findall(r'\w+', pred_clean)
        actual_words = re.findall(r'\w+', actual_clean)
        
        best_similarity = 0.0
        best_description = ""
        
        for p_word in pred_words:
            for a_word in actual_words:
                for action, similarities in self.action_similarity.items():
                    if p_word in similarities and a_word in similarities:
                        similarity = similarities[a_word]
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_description = f"Action similarity: '{p_word}' vs '{a_word}'"
        
        if best_similarity >= 0.8:
            return PremiseScore("action", best_similarity, "High action similarity", best_description)
        elif best_similarity >= 0.6:
            return PremiseScore("action", best_similarity, "Moderate action similarity", best_description)
        
        # Category-based matching
        violent_actions = ['crash', 'collide', 'hit', 'smash', 'bump', 'accident']
        gentle_actions = ['park', 'stop', 'stand', 'wait', 'visit']
        movement_actions = ['move', 'drive', 'travel', 'go']
        
        pred_violent = any(action in pred_clean for action in violent_actions)
        actual_violent = any(action in actual_clean for action in violent_actions)
        pred_gentle = any(action in pred_clean for action in gentle_actions)
        actual_gentle = any(action in actual_clean for action in gentle_actions)
        
        if pred_violent and actual_violent:
            return PremiseScore("action", 0.9, "Same action category (violent)", "Different specific violent action")
        if pred_gentle and actual_gentle:
            return PremiseScore("action", 0.9, "Same action category (gentle)", "Different specific gentle action")
        if (pred_violent and actual_gentle) or (pred_gentle and actual_violent):
            return PremiseScore("action", 0.3, "Opposite action categories", "Violent vs gentle action mismatch")
        
        return PremiseScore("action", 0.5, "Different action specifics", "Different action description")
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse various date formats"""
        date_str = date_str.lower().strip()
        
        # Current year as default
        current_year = datetime.now().year
        
        # Try different date patterns
        patterns = [
            (r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: datetime(int(m.group(3)), int(m.group(1)), int(m.group(2)))),
            (r'(\d{1,2})-(\d{1,2})-(\d{4})', lambda m: datetime(int(m.group(3)), int(m.group(1)), int(m.group(2)))),
            (r'(\w+) (\d{1,2}),? (\d{4})', self._parse_month_date),
            (r'(\d{4})', lambda m: datetime(int(m.group(1)), 6, 15)),  # Middle of year
        ]
        
        for pattern, parser in patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    return parser(match)
                except:
                    continue
        
        # If no specific date found, return a date based on context
        if '2024' in date_str:
            return datetime(2024, 6, 15)
        elif '2023' in date_str:
            return datetime(2023, 6, 15)
        elif '2025' in date_str:
            return datetime(2025, 6, 15)
        
        # Default to current date
        return datetime.now()
    
    def _parse_month_date(self, match) -> datetime:
        """Parse month name date format"""
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
            'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        month_name = match.group(1).lower()
        day = int(match.group(2))
        year = int(match.group(3))
        month = months.get(month_name, 1)
        return datetime(year, month, min(day, 28))  # Avoid day overflow
    
    def analyze_claim(self, prediction: Dict, actual: Dict) -> Dict[str, Any]:
        """Main method to analyze claim accuracy"""
        premise_scores = []
        
        # Calculate scores for each premise
        if 'time' in prediction and 'time' in actual:
            premise_scores.append(self.calculate_time_score(prediction['time'], actual['time']))
        
        if 'location' in prediction and 'location' in actual:
            premise_scores.append(self.calculate_location_score(prediction['location'], actual['location']))
        
        if 'subject' in prediction and 'subject' in actual:
            premise_scores.append(self.calculate_entity_score(prediction['subject'], actual['subject'], 'subject'))
        
        if 'object' in prediction and 'object' in actual:
            premise_scores.append(self.calculate_entity_score(prediction['object'], actual['object'], 'object'))
        
        if 'action' in prediction and 'action' in actual:
            premise_scores.append(self.calculate_action_score(prediction['action'], actual['action']))
        
        # Calculate overall CFAS
        cfas_score = 1.0
        for premise in premise_scores:
            cfas_score *= premise.score
        
        # Generate recommendations
        recommendations = self._generate_recommendations(premise_scores, cfas_score)
        
        return {
            'medusa_score': round(cfas_score, 3),
            'premise_scores': {ps.name: {'score': ps.score, 'description': ps.description, 'deviation': ps.deviation} for ps in premise_scores},
            'distance_from_perfect': round(1.0 - cfas_score, 3),
            'weakest_premises': self._get_weakest_premises(premise_scores),
            'recommendations': recommendations,
            'interpretation': self._interpret_score(cfas_score)
        }
    
    def _generate_recommendations(self, premise_scores: List[PremiseScore], overall_score: float) -> List[str]:
        """Generate improvement recommendations based on scoring gaps"""
        recommendations = []
        
        for premise in premise_scores:
            if premise.score < 0.8:
                recommendations.append(
                    f"Improve {premise.name} accuracy: {premise.deviation} (current score: {premise.score})"
                )
        
        if overall_score < 0.7:
            recommendations.append("Consider making predictions more specific and verifiable")
        
        if len([ps for ps in premise_scores if ps.score < 0.5]) > 0:
            recommendations.append("Review core premises - significant mismatches detected")
        
        return recommendations
    
    def _get_weakest_premises(self, premise_scores: List[PremiseScore]) -> List[str]:
        """Identify premises with lowest scores"""
        weak_premises = [ps for ps in premise_scores if ps.score < 0.8]
        return [f"{ps.name} (score: {ps.score})" for ps in sorted(weak_premises, key=lambda x: x.score)]
    
    def _interpret_score(self, score: float) -> str:
        """Provide interpretation of the final score"""
        if score >= 0.95:
            return "Excellent accuracy - claim is highly precise and correct"
        elif score >= 0.8:
            return "Very good accuracy - minor deviations from prediction"
        elif score >= 0.6:
            return "Good accuracy - moderate deviations but core claim correct"
        elif score >= 0.4:
            return "Moderate accuracy - several premises have notable deviations"
        elif score >= 0.2:
            return "Poor accuracy - major deviations in multiple premises"
        else:
            return "Very poor accuracy - fundamental mismatches with reality"

def load_scenarios_from_yaml(file_path: str) -> List[Dict]:
    """Load scenarios from YAML file"""
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data.get('scenarios', [])
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return []

def main():
    scorer = ClaimsForecastScorer()
    
    # Load scenarios from YAML file
    yaml_file = "task.yml"  # Change this to your YAML file path
    scenarios = load_scenarios_from_yaml(yaml_file)
    
    if not scenarios:
        print("No scenarios found in YAML file. Using default examples.")
        # Fallback to default scenarios if YAML file is not found
        scenarios = [
            {
                'name': 'Perfect Prediction',
                'prediction': {
                    'time': 'April 10, 2024',
                    'location': 'Eiffel Tower, Paris',
                    'subject': 'a car',
                    'action': 'collide',
                    'object': 'the Eiffel Tower'
                },
                'actual': {
                    'time': 'April 10, 2024',
                    'location': 'Eiffel Tower, Paris',
                    'subject': 'a red Toyota Camry',
                    'action': 'collide',
                    'object': 'Eiffel Tower'
                }
            }
        ]
    
    print(f"Loaded {len(scenarios)} scenario(s) from {yaml_file}")
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"{'='*60}")
        
        result = scorer.analyze_claim(scenario['prediction'], scenario['actual'])
        
        print(f"MEDUSA Score: {result['medusa_score']:.3f}")
        print(f"Distance from 1.0: {result['distance_from_perfect']:.3f}")
        print(f"Interpretation: {result['interpretation']}")
        
        print("\nPremise Scores:")
        for premise, details in result['premise_scores'].items():
            print(f"  - {premise:8} | Score: {details['score']:.3f} | {details['description']:30} | {details['deviation']}")
        
        if result['weakest_premises']:
            print(f"\nWeakest Premises: {', '.join(result['weakest_premises'])}")
        
        if result['recommendations']:
            print("\nRecommendations:")
            for rec in result['recommendations']:
                print(f"  • {rec}")

if __name__ == "__main__":
    main()

"""
Loaded 5 scenario(s) from task.yml

============================================================
SCENARIO: Toy Car vs Real Car
============================================================
MEDUSA Score: 0.730
Distance from 1.0: 0.270
Interpretation: Good accuracy - moderate deviations but core claim correct

Premise Scores:
  - time     | Score: 0.730 | Generic time accuracy          | Generic time reference off by 100 days
  - location | Score: 1.000 | Location hierarchy match       | No deviation
  - subject  | Score: 1.000 | High conceptual similarity     | Word similarity: 'car' vs 'car' in category 'vehicle'
  - object   | Score: 1.000 | High conceptual similarity     | Word similarity: 'louvre' vs 'tower' in category 'monument'
  - action   | Score: 1.000 | High action similarity         | Action similarity: 'collide' vs 'crash'

Weakest Premises: time (score: 0.73)

Recommendations:
  • Improve time accuracy: Generic time reference off by 100 days (current score: 0.73)

============================================================
SCENARIO: Month Accuracy Test
============================================================
MEDUSA Score: 0.300
Distance from 1.0: 0.700
Interpretation: Poor accuracy - major deviations in multiple premises

Premise Scores:
  - time     | Score: 0.300 | Month accuracy                 | Wrong month
  - location | Score: 1.000 | Location contains match        | No deviation
  - subject  | Score: 1.000 | High conceptual similarity     | Word similarity: 'vehicle' vs 'car' in category 'car'
  - object   | Score: 1.000 | High conceptual similarity     | Word similarity: 'monument' vs 'tower' in category 'monument'
  - action   | Score: 1.000 | High action similarity         | Action similarity: 'bump' vs 'collide'

Weakest Premises: time (score: 0.3)

Recommendations:
  • Improve time accuracy: Wrong month (current score: 0.3)
  • Consider making predictions more specific and verifiable
  • Review core premises - significant mismatches detected

============================================================
SCENARIO: Same Country Different City
============================================================
MEDUSA Score: 0.500
Distance from 1.0: 0.500
Interpretation: Moderate accuracy - several premises have notable deviations

Premise Scores:
  - time     | Score: 1.000 | Year accuracy                  | Year accuracy
  - location | Score: 1.000 | Location hierarchy match       | No deviation
  - subject  | Score: 1.000 | High conceptual similarity     | Word similarity: 'truck' vs 'car' in category 'vehicle'
  - object   | Score: 1.000 | High conceptual similarity     | Word similarity: 'building' vs 'monument' in category 'monument'
  - action   | Score: 0.500 | Different action specifics     | Different action description

Weakest Premises: action (score: 0.5)

Recommendations:
  • Improve action accuracy: Different action description (current score: 0.5)
  • Consider making predictions more specific and verifiable

============================================================
SCENARIO: Perfect Match
============================================================
MEDUSA Score: 1.000
Distance from 1.0: 0.000
Interpretation: Excellent accuracy - claim is highly precise and correct

Premise Scores:
  - time     | Score: 1.000 | Exact time match               | No deviation
  - location | Score: 1.000 | Exact location match           | No deviation
  - subject  | Score: 1.000 | High conceptual similarity     | Word similarity: 'car' vs 'red' in category 'car'
  - object   | Score: 1.000 | object contains match          | No deviation
  - action   | Score: 1.000 | Exact action match             | No deviation

============================================================
SCENARIO: Vague but Correct
============================================================
MEDUSA Score: 0.950
Distance from 1.0: 0.050
Interpretation: Excellent accuracy - claim is highly precise and correct

Premise Scores:
  - time     | Score: 1.000 | Year accuracy                  | Year accuracy
  - location | Score: 1.000 | Location hierarchy match       | No deviation
  - subject  | Score: 1.000 | High conceptual similarity     | Word similarity: 'vehicle' vs 'car' in category 'car'
  - object   | Score: 1.000 | High conceptual similarity     | Word similarity: 'landmark' vs 'tower' in category 'monument'
  - action   | Score: 0.950 | High action similarity         | Action similarity: 'incident' vs 'crash'

============================================================
SCENARIO: Utterly Off
============================================================
MEDUSA Score: 0.019
Distance from 1.0: 0.981
Interpretation: Very poor accuracy - fundamental mismatches with reality

Premise Scores:
  - time     | Score: 0.500 | Generic time accuracy          | Generic time reference off by 1395 days
  - location | Score: 0.300 | Different location             | Completely different location
  - subject  | Score: 0.500 | Different specific entity      | Predicted 'balloon' vs actual 'buffalo'
  - object   | Score: 0.500 | Different specific entity      | Predicted 'object misplacement' vs actual 'animal'
  - action   | Score: 0.500 | Different action specifics     | Different action description

Weakest Premises: location (score: 0.3), time (score: 0.5), subject (score: 0.5), object (score: 0.5), action (score: 0.5)

Recommendations:
  • Improve time accuracy: Generic time reference off by 1395 days (current score: 0.5)
  • Improve location accuracy: Completely different location (current score: 0.3)
  • Improve subject accuracy: Predicted 'balloon' vs actual 'buffalo' (current score: 0.5)
  • Improve object accuracy: Predicted 'object misplacement' vs actual 'animal' (current score: 0.5)
  • Improve action accuracy: Different action description (current score: 0.5)
  • Consider making predictions more specific and verifiable
  • Review core premises - significant mismatches detected
    
"""