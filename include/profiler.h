/**
 * filename: profiler.h
 * contents: this file contains the macros used to time sections of the code
 */

#ifndef PROFILER_H
#define PROFILER_H

#include "util.h"

#include <map>
#include <ctime>
#include <string>
#include <fstream>

using namespace std;

/**
 * class used to time functions
 */
class Profiler {

  public:

    static Profiler& getInstance() {
      static Profiler prof;

      return prof;
    }

    void startEvent(const string event) {
      // crete time stamp for this event
      times_[event] = clock();
    }

    void stopEvent(const string event) {
      // calculate time usage and print
      clock_t end = clock();
      if(times_.find(event) == times_.end()) {
        LOG_ERROR("Event \"" + event + "\" does not exist");
      }
    
      double elapsedTime = double(end - times_[event]) / CLOCKS_PER_SEC;
      fp_ << fixed << "\"" << event << "\" took " << elapsedTime << " seconds" << endl;
      
      // save time
      times_[event] = elapsedTime;
    }

    double getTime(const string event) {
      return times_[event];
    }

  private:
    // should never be instantiated
    Profiler() { 
      cout.precision(6); 
      rank_ = MPI::COMM_WORLD.Get_rank();
      fp_.open("log/slave_" + to_string(rank_) + ".log");
    }

    ~Profiler() { fp_.close(); }

    map<string, clock_t> times_;
    int rank_;
    ofstream fp_;
};

// profiler macros
#define PROFILER_START_EVENT(string) { Profiler::getInstance().startEvent(string); }

#define PROFILER_STOP_EVENT(string) { Profiler::getInstance().stopEvent(string); }

#endif // PROFILER_H
