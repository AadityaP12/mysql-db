from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
import logging

from app.ml.model import disease_prediction_model
from app.schemas.ml import RiskPrediction, OutbreakPrediction, DiseaseType, RiskLevel
from app.core.utils import get_risk_level, calculate_distance
from app.db.database import firestore_service, FirestoreCollections

logger = logging.getLogger(__name__)


class RiskPredictor:
    """
    Service class for making risk predictions and managing prediction workflows
    """
    
    def __init__(self):
        self.model = disease_prediction_model
    
    async def predict_risk(self, prediction_input: Dict[str, Any]) -> RiskPrediction:
        """
        Generate risk prediction for a specific location and conditions
        """
        try:
            # Prepare features for the ML model
            features = self.model.prepare_features(prediction_input)
            
            # Get risk score from model
            risk_score = self.model.predict_risk_score(features)
            
            # Get outbreak predictions
            outbreak_data = self.model.predict_outbreak_probability(features)
            
            # Determine risk level and color
            risk_info = get_risk_level(risk_score)
            
            # Get primary risk factors
            risk_factors = self.model.get_risk_factors(features, risk_score)
            
            # Map predicted diseases to enum types
            predicted_diseases = []
            for disease in outbreak_data.get('predicted_diseases', []):
                try:
                    predicted_diseases.append(DiseaseType(disease))
                except ValueError:
                    # Handle unknown diseases
                    predicted_diseases.append(DiseaseType.GENERAL_WATERBORNE)
            
            # Estimate population at risk
            population_at_risk = await self._estimate_population_at_risk(
                prediction_input.get('location', {}),
                risk_score
            )
            
            # Create risk prediction
            risk_prediction = RiskPrediction(
                location=prediction_input['location'],
                risk_score=risk_score,
                risk_level=RiskLevel(risk_info['level'].lower()),
                risk_color=risk_info['color'],
                confidence=outbreak_data.get('outbreak_probability', 0.7),
                predicted_diseases=predicted_diseases,
                primary_risk_factors=risk_factors,
                population_at_risk=population_at_risk,
                prediction_timestamp=datetime.now(timezone.utc),
                model_version=getattr(self.model, 'version', 'v1.0')
            )
            
            # Store prediction for historical tracking
            await self._store_prediction(risk_prediction)
            
            return risk_prediction
            
        except Exception as e:
            logger.error(f"Error generating risk prediction: {str(e)}")
            # Return default moderate risk prediction
            return await self._create_fallback_prediction(prediction_input)
    
    async def predict_outbreak(self, region_id: str, prediction_data: Dict[str, Any]) -> OutbreakPrediction:
        """
        Generate outbreak prediction for a specific region
        """
        try:
            # Prepare features
            features = self.model.prepare_features(prediction_data)
            
            # Get outbreak predictions
            outbreak_data = self.model.predict_outbreak_probability(features)
            outbreak_probability = outbreak_data.get('outbreak_probability', 0.3)
            
            # Estimate expected cases based on population and risk
            population = prediction_data.get('population_density', 100) * 10  # Rough estimate
            expected_cases = int(population * outbreak_probability * 0.1)  # 10% attack rate
            
            # Estimate peak time (simplified model)
            peak_days = 7 + (14 * (1 - outbreak_probability))  # 7-21 days
            peak_time = datetime.now(timezone.utc) + timedelta(days=int(peak_days))
            
            # Determine primary disease type
            predicted_diseases = outbreak_data.get('predicted_diseases', [])
            primary_disease = predicted_diseases[0] if predicted_diseases else 'general_waterborne'
            
            try:
                disease_type = DiseaseType(primary_disease)
            except ValueError:
                disease_type = DiseaseType.GENERAL_WATERBORNE
            
            # Generate preventive measures
            preventive_measures = self._generate_preventive_measures(
                outbreak_data.get('disease_probabilities', {}),
                outbreak_probability
            )
            
            # Calculate resource requirements
            resource_requirements = self._calculate_resource_requirements(
                expected_cases,
                disease_type,
                outbreak_probability
            )
            
            # Determine affected radius
            affected_radius = 2.0 + (outbreak_probability * 8.0)  # 2-10 km based on risk
            
            outbreak_prediction = OutbreakPrediction(
                region_id=region_id,
                outbreak_probability=outbreak_probability,
                expected_cases=expected_cases,
                peak_time_estimate=peak_time,
                disease_type=disease_type,
                affected_radius_km=affected_radius,
                preventive_measures=preventive_measures,
                resource_requirements=resource_requirements,
                prediction_accuracy=0.75  # Model confidence estimate
            )
            
            # Store outbreak prediction
            await self._store_outbreak_prediction(outbreak_prediction)
            
            return outbreak_prediction
            
        except Exception as e:
            logger.error(f"Error generating outbreak prediction: {str(e)}")
            # Return fallback prediction
            return await self._create_fallback_outbreak_prediction(region_id)
    
    async def batch_risk_assessment(self, locations: List[Dict[str, float]]) -> List[RiskPrediction]:
        """
        Perform risk assessment for multiple locations
        """
        predictions = []
        
        for location in locations:
            try:
                # Get recent data for this location
                location_data = await self._get_location_data(location)
                
                # Generate prediction
                prediction = await self.predict_risk(location_data)
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Error in batch assessment for location {location}: {str(e)}")
                # Add fallback prediction
                fallback = await self._create_fallback_prediction({'location': location})
                predictions.append(fallback)
        
        return predictions
    
    async def get_regional_risk_map(self, region_bounds: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate risk map data for a geographic region
        """
        try:
            # Get all recent water quality and health data in region
            water_quality_data = await firestore_service.query_collection(
                FirestoreCollections.WATER_QUALITY,
                filters=[
                    ("collection_time", ">=", datetime.now(timezone.utc) - timedelta(days=7))
                ],
                limit=1000
            )
            
            health_data = await firestore_service.query_collection(
                FirestoreCollections.HEALTH_DATA,
                filters=[
                    ("report_time", ">=", datetime.now(timezone.utc) - timedelta(days=14))
                ],
                limit=1000
            )
            
            # Create risk assessment for grid points in region
            risk_points = []
            
            # Generate grid of assessment points
            lat_min, lat_max = region_bounds.get('lat_min', 21.0), region_bounds.get('lat_max', 30.0)
            lon_min, lon_max = region_bounds.get('lon_min', 87.0), region_bounds.get('lon_max', 98.0)
            
            grid_size = 0.1  # ~10km grid
            for lat in np.arange(lat_min, lat_max, grid_size):
                for lon in np.arange(lon_min, lon_max, grid_size):
                    location = {'latitude': lat, 'longitude': lon}
                    
                    # Aggregate nearby data
                    nearby_data = self._aggregate_nearby_data(
                        location, water_quality_data, health_data, radius_km=10
                    )
                    
                    if nearby_data:  # Only assess if there's data nearby
                        prediction = await self.predict_risk({
                            'location': location,
                            **nearby_data
                        })
                        
                        risk_points.append({
                            'location': location,
                            'risk_score': prediction.risk_score,
                            'risk_level': prediction.risk_level.value,
                            'risk_color': prediction.risk_color
                        })
            
            # Calculate regional statistics
            risk_scores = [p['risk_score'] for p in risk_points]
            regional_stats = {
                'average_risk': sum(risk_scores) / len(risk_scores) if risk_scores else 0,
                'high_risk_areas': len([p for p in risk_points if p['risk_score'] > 0.7]),
                'medium_risk_areas': len([p for p in risk_points if 0.3 <= p['risk_score'] <= 0.7]),
                'low_risk_areas': len([p for p in risk_points if p['risk_score'] < 0.3]),
                'total_assessed_points': len(risk_points)
            }
            
            return {
                'risk_map_data': risk_points,
                'regional_statistics': regional_stats,
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'coverage_area': region_bounds
            }
            
        except Exception as e:
            logger.error(f"Error generating regional risk map: {str(e)}")
            return {
                'risk_map_data': [],
                'regional_statistics': {},
                'error': str(e)
            }
    
    async def _estimate_population_at_risk(self, location: Dict[str, float], risk_score: float) -> Optional[int]:
        """
        Estimate population at risk based on location and risk score
        """
        try:
            # Get nearby households
            households = await firestore_service.query_collection(
                FirestoreCollections.HOUSEHOLDS,
                limit=500
            )
            
            population_count = 0
            risk_radius = 5.0 * risk_score  # Risk radius based on score
            
            for household in households:
                hh_location = household.get('location', {})
                if 'latitude' in hh_location and 'longitude' in hh_location:
                    distance = calculate_distance(
                        location['latitude'], location['longitude'],
                        hh_location['latitude'], hh_location['longitude']
                    )
                    
                    if distance <= risk_radius:
                        population_count += household.get('total_members', 4)
            
            return population_count if population_count > 0 else None
            
        except Exception as e:
            logger.error(f"Error estimating population at risk: {str(e)}")
            return None
    
    async def _store_prediction(self, prediction: RiskPrediction):
        """
        Store prediction in database for historical tracking
        """
        try:
            prediction_data = prediction.dict()
            prediction_data['id'] = f"PRED_{int(datetime.now().timestamp())}"
            
            await firestore_service.create_document(
                FirestoreCollections.ML_PREDICTIONS,
                prediction_data
            )
            
        except Exception as e:
            logger.error(f"Error storing prediction: {str(e)}")
    
    async def _store_outbreak_prediction(self, prediction: OutbreakPrediction):
        """
        Store outbreak prediction in database
        """
        try:
            prediction_data = prediction.dict()
            prediction_data['id'] = f"OUTBREAK_PRED_{int(datetime.now().timestamp())}"
            prediction_data['prediction_type'] = 'outbreak'
            
            await firestore_service.create_document(
                FirestoreCollections.ML_PREDICTIONS,
                prediction_data
            )
            
        except Exception as e:
            logger.error(f"Error storing outbreak prediction: {str(e)}")
    
    async def _get_location_data(self, location: Dict[str, float]) -> Dict[str, Any]:
        """
        Gather relevant data for a specific location
        """
        try:
            # Get recent water quality data nearby
            water_quality_data = await firestore_service.query_collection(
                FirestoreCollections.WATER_QUALITY,
                filters=[
                    ("collection_time", ">=", datetime.now(timezone.utc) - timedelta(days=3))
                ],
                limit=100
            )
            
            # Get recent health data nearby
            health_data = await firestore_service.query_collection(
                FirestoreCollections.HEALTH_DATA,
                filters=[
                    ("report_time", ">=", datetime.now(timezone.utc) - timedelta(days=7))
                ],
                limit=100
            )
            
            # Aggregate data for this location
            return self._aggregate_nearby_data(location, water_quality_data, health_data, radius_km=5)
            
        except Exception as e:
            logger.error(f"Error getting location data: {str(e)}")
            return {'location': location}
    
    def _aggregate_nearby_data(
        self, 
        location: Dict[str, float], 
        water_data: List[Dict], 
        health_data: List[Dict], 
        radius_km: float = 5.0
    ) -> Dict[str, Any]:
        """
        Aggregate data from nearby locations
        """
        try:
            lat, lon = location['latitude'], location['longitude']
            
            # Filter nearby water quality data
            nearby_water = []
            for record in water_data:
                record_location = record.get('location', {})
                if 'latitude' in record_location:
                    distance = calculate_distance(
                        lat, lon,
                        record_location['latitude'],
                        record_location['longitude']
                    )
                    if distance <= radius_km:
                        nearby_water.append(record)
            
            # Filter nearby health data
            nearby_health = []
            for record in health_data:
                record_location = record.get('location', {})
                if 'latitude' in record_location:
                    distance = calculate_distance(
                        lat, lon,
                        record_location['latitude'],
                        record_location['longitude']
                    )
                    if distance <= radius_km:
                        nearby_health.append(record)
            
            # Aggregate water quality metrics
            if nearby_water:
                avg_ph = sum(r.get('ph_level', 7.0) for r in nearby_water) / len(nearby_water)
                avg_turbidity = sum(r.get('turbidity', 0.0) for r in nearby_water) / len(nearby_water)
                avg_chlorine = sum(r.get('residual_chlorine', 0.0) for r in nearby_water) / len(nearby_water)
                avg_temp = sum(r.get('temperature', 25.0) for r in nearby_water) / len(nearby_water)
                
                water_quality_data = {
                    'ph_level': avg_ph,
                    'turbidity': avg_turbidity,
                    'residual_chlorine': avg_chlorine,
                    'temperature': avg_temp
                }
            else:
                # Default values if no nearby data
                water_quality_data = {
                    'ph_level': 7.0,
                    'turbidity': 1.0,
                    'residual_chlorine': 0.2,
                    'temperature': 25.0
                }
            
            # Aggregate symptom counts
            symptom_counts = {}
            for record in nearby_health:
                for symptom in record.get('symptoms', []):
                    symptom_counts[symptom] = symptom_counts.get(symptom, 0) + 1
            
            return {
                'location': location,
                'water_quality_data': water_quality_data,
                'health_symptoms_count': symptom_counts,
                'population_density': len(nearby_health) * 10,  # Rough estimate
                'sanitation_score': 5.0,  # Default score
                'nearby_data_points': {
                    'water_quality': len(nearby_water),
                    'health_reports': len(nearby_health)
                }
            }
            
        except Exception as e:
            logger.error(f"Error aggregating nearby data: {str(e)}")
            return {'location': location}
    
    def _generate_preventive_measures(self, disease_probabilities: Dict[str, float], outbreak_probability: float) -> List[str]:
        """
        Generate context-appropriate preventive measures
        """
        measures = []
        
        if outbreak_probability > 0.7:
            measures.extend([
                "Immediate water source disinfection required",
                "Boil all water for at least 10 minutes before consumption",
                "Implement strict hand hygiene protocols",
                "Increase health surveillance in affected areas"
            ])
        elif outbreak_probability > 0.4:
            measures.extend([
                "Boil water before consumption as a precaution",
                "Increase water quality monitoring frequency",
                "Conduct health awareness campaigns",
                "Ensure proper waste disposal"
            ])
        else:
            measures.extend([
                "Maintain good water storage practices",
                "Regular hand washing with soap",
                "Proper food hygiene practices"
            ])
        
        # Disease-specific measures
        if 'cholera' in disease_probabilities:
            measures.append("Immediate reporting of severe diarrhea cases")
        if 'typhoid' in disease_probabilities:
            measures.append("Enhanced personal hygiene measures")
        if 'hepatitis_a' in disease_probabilities:
            measures.append("Proper sanitation and sewage management")
        
        return measures
    
    def _calculate_resource_requirements(self, expected_cases: int, disease_type: DiseaseType, probability: float) -> Dict[str, int]:
        """
        Calculate required resources based on outbreak prediction
        """
        base_multiplier = max(1, int(probability * 10))
        
        resources = {
            "ors_packets": expected_cases * 5 * base_multiplier,
            "water_purification_tablets": expected_cases * 20 * base_multiplier,
            "health_workers": max(1, expected_cases // 10),
            "testing_kits": expected_cases * 2,
            "antibiotics": expected_cases * 3 if disease_type in [DiseaseType.CHOLERA, DiseaseType.TYPHOID] else 0,
            "isolation_beds": max(1, expected_cases // 5) if probability > 0.6 else 0
        }
        
        return resources
    
    async def _create_fallback_prediction(self, prediction_input: Dict[str, Any]) -> RiskPrediction:
        """
        Create fallback prediction when main prediction fails
        """
        return RiskPrediction(
            location=prediction_input.get('location', {'latitude': 0, 'longitude': 0}),
            risk_score=0.5,
            risk_level=RiskLevel.MEDIUM,
            risk_color="YELLOW",
            confidence=0.6,
            predicted_diseases=[DiseaseType.GENERAL_WATERBORNE],
            primary_risk_factors=["Unable to determine specific risk factors"],
            population_at_risk=None,
            prediction_timestamp=datetime.now(timezone.utc),
            model_version="fallback_v1.0"
        )
    
    async def _create_fallback_outbreak_prediction(self, region_id: str) -> OutbreakPrediction:
        """
        Create fallback outbreak prediction
        """
        return OutbreakPrediction(
            region_id=region_id,
            outbreak_probability=0.3,
            expected_cases=5,
            peak_time_estimate=datetime.now(timezone.utc) + timedelta(days=14),
            disease_type=DiseaseType.GENERAL_WATERBORNE,
            affected_radius_km=3.0,
            preventive_measures=[
                "Boil water before consumption",
                "Maintain proper hygiene",
                "Monitor water quality"
            ],
            resource_requirements={
                "ors_packets": 25,
                "water_purification_tablets": 100,
                "health_workers": 1
            },
            prediction_accuracy=0.6
        )


# Import numpy for grid generation
import numpy as np