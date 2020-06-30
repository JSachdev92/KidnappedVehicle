/**
 * particle_filter.cpp
 *
 * Created on: June 29, 2020
 * Author: Jayant Sachdev
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  vector<double> weights;
  num_particles =100;  // TODO: Set the number of particles
  particles.resize(num_particles);
  weights.resize(num_particles);
  std::default_random_engine gen;

  
    // This line creates a normal (Gaussian) distribution for x, y and Theta
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  for (int i = 0; i < num_particles; ++i) {
    
    // Sample from these normal distributions: 
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  std::default_random_engine gen;
  
  //Normal distributions for sensor noise
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);
  
  //adjust all particles based on movement
  for (int i = 0; i< num_particles; ++i) {
    //protect against division by zero
    if(fabs(yaw_rate)>0.00001){
      particles[i].x +=  velocity/yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta +=  yaw_rate*delta_t;
    }else{ // we can use zero yaw rate equations
      particles[i].x += velocity*delta_t*cos(particles[i].theta);
      particles[i].y += velocity*delta_t*sin(particles[i].theta);
    }
    
    //add random gaussian noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
    
  }
          
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (unsigned int i = 0; i < observations.size(); ++i) { // For each observation

    // Initialize min distance as a really big number.
    double minDistance = std::numeric_limits<double>::max();

    for (unsigned j = 0; j < predicted.size(); ++j ) { // For each predition.
   
      double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

      // If the "distance" is less than min, stored the id and update min.
      if ( distance < minDistance ) {
        minDistance = distance;
        observations[i].id = predicted[j].id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];
  double gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);
  vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list;
  
  for(auto& p: particles){
    
    p.weight = 1.0;
    double p_x = p.x;
    double p_y = p.y;
	double p_theta = p.theta;
    // step 1: collect valid landmarks
    vector<LandmarkObs> predictions;
    for(unsigned int j=0; j<landmarks.size(); ++j){
      //distance between particle and landmark
      double landmark_dist = dist(p_x, p_y, landmarks[j].x_f, landmarks[j].y_f);     
      // select landmarks in sensor range
      
      if(landmark_dist < sensor_range){              
        predictions.push_back(LandmarkObs{landmarks[j].id_i, landmarks[j].x_f, landmarks[j].y_f});
      } 
    }

    // step 2: convert observations coordinates from vehicle to map
    vector<LandmarkObs> obs_m;
    LandmarkObs temp;
    
    double cos_theta = cos(p_theta);
    double sin_theta = sin(p_theta);

    for(unsigned int k=0; k<observations.size(); ++k){            
      temp.x = p_x + cos_theta*observations[k].x - sin_theta*observations[k].y;           
      temp.y = p_y + sin_theta*observations[k].x + cos_theta*observations[k].y;           
      obs_m.push_back(temp);
    }

    // step 3: find landmark index for each observation
    dataAssociation(predictions, obs_m);

    // step 4: compute the particle's weight:
    // see equation this link:
    double weight_upd = 1.0;
       for(unsigned int z=0; z<obs_m.size(); ++z){            
         //save observed landmark to nearist single landmark map type based on data association and use for calculation of weights
         
         //Map index starts at 1, adjust to 0
         Map::single_landmark_s obs_l = map_landmarks.landmark_list.at(obs_m[z].id-1);
         //Multivariate-Gaussian probability      
         double x_term = pow(obs_m[z].x - obs_l.x_f, 2) / (2*pow(sig_x, 2));
         double y_term = pow(obs_m[z].y - obs_l.y_f, 2) / (2*pow(sig_y, 2));
         double w = gauss_norm*exp(-(x_term + y_term));
//          std::cout << "Obs_measured x: " << obs_m[z].x << " Obs_measured y:  " << obs_m[z].y << std::endl;
//          std::cout << "Obs_landmark x: " << obs_l.x_f << " Obs_landmark y:  " << obs_l.y_f << std::endl;
//          std::cout << "x term: " << x_term << " y term: " << y_term << " gaussian norm: " << gauss_norm << std::endl;
//          std::cout << "weight" << z << ": " << w << std::endl;
            weight_upd *= w;
       }   
    p.weight = weight_upd;
    weights.push_back(weight_upd);

  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  //std::cout << "particle weights resampling..." << std::endl;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> distribution( weights.begin(), weights.end()) ;
  weights.clear();
  // resample particles
  std::vector<Particle> resampled_particles;
  resampled_particles.resize(num_particles);
  for(int i = 0; i < num_particles; ++i){
      int idx = distribution(gen);
      resampled_particles[i] = particles[idx];
    }

  // assign resampled_particles to particles
  particles = resampled_particles;
  
  weights.clear();
    
    
 // std::cout << "particles resampled" << std::endl;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}