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

struct timer{

  std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;
  std::string name;

  void begin(std::string name = "unnamed"){ this->name = name; start = std::chrono::high_resolution_clock::now();}
  float end(){
		float dt = end_silent();
    std::cout << "Time taken by " << name << " is " << dt  << " seconds\n";
		return dt;
  }
	float end_silent(){
		stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		return (float)duration.count()/1.0e6;
	}
};

#endif
