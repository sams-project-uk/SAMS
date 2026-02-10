/*
 *    Copyright 2024 C.S.Brady & H.Ratcliffe

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#ifndef TIMER_H
#define TIMER_H
#include <chrono>
#include <string>
#include <iostream>
#include "mpiManager.h"

struct timer{
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::string name;
  double elapsed;/** Elapsed time in microseconds */
  bool isPaused = true;
  bool hasEverRun = false;

  void updateElapsed(){
    hasEverRun = true;
    if (isPaused) return;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
    elapsed += (double)duration.count();
  }

  public:

  timer(std::string name) : name(name), elapsed(0.0), isPaused(true) {}

  timer() : name("unnamed"), elapsed(0.0), isPaused(true) {}

  /**
   * Check if the timer has ever been run
   */
  bool everRun() const {
    return hasEverRun;
  }

  /**
   * Begin a new timing session
   */
  void begin(std::string name = "unnamed"){ 
    this->name = name; 
    isPaused = false;
    start = std::chrono::high_resolution_clock::now();
    elapsed = 0.0;
  }

  /**
   * Begin a new timing session in a paused state
   */
  void begin_paused(std::string name = "unnamed"){
    this->name = name; 
    elapsed = 0.0;
    isPaused = true;
  }

  /**
   * Toggle pause state
   */
  void toggle(){
    if (!isPaused){
      //Pause the timer
      updateElapsed();
      isPaused = true;
    } else {
      //Resume the timer
      start = std::chrono::high_resolution_clock::now();
      isPaused = false;
    }
  }

  /**
   * End the current timing session and print the result
   */
  float end(){
		float dt = end_silent();
    SAMS::cout << "Time taken by " << name << " is " << dt  << " seconds\n";
		return dt;
  }

  /**
   * End the current timing session and return the elapsed time in seconds
   */
	double end_silent(){
    updateElapsed();
    return elapsed*1.0e-6;
	}
};

#endif
