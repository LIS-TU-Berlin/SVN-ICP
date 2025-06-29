/*  ------------------------------------------------------------------
Copyright (c) 2020-2025 Shiping Ma and Haoming Zhang
    email: shiping.ma@tu-berlin.de and haoming.zhang@rwth-aachen.de

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

/**
 * @file    SignalSmoother.h
 * @brief   smoother to smooth the estimated variances
 * @author  Shiping Ma*
 * @author  Haoming Zhang*
 * @date    June 22, 2025
 */

#ifndef SIGNALSMOOTHER_H
#define SIGNALSMOOTHER_H
#pragma once

#include <deque>
#include "data/DataTypes.h"



namespace svnicp {

  /**
  * Max. siding window smoother to smooth the estimated noise parameters
  * => currently not used
  */
  class MaxSlidingWindow {
  private:
    std::deque<std::pair<double, int> > window_;
    int windowSize_;
    int index_ = 0;

  public:
    explicit MaxSlidingWindow(const int size) : windowSize_(size) {
    }

    double filter(double value) {
      // Remove values that are outside the window
      while (!window_.empty() && window_.front().second <= index_ - windowSize_) {
        window_.pop_front();
      }

      // Remove values smaller than the current one
      while (!window_.empty() && window_.back().first <= value) {
        window_.pop_back();
      }

      // Add the new value
      window_.emplace_back(value, index_++);

      // The maximum value is at the front
      return window_.front().first;
    }
  };


}


#endif //SIGNALSMOOTHER_H
