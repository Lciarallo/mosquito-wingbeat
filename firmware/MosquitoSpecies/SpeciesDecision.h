#ifndef MOSQUITO_SPECIES_DECISION_H
#define MOSQUITO_SPECIES_DECISION_H
#include <stdint.h>

namespace mosquito {
// A missing/uncertain candidate occupies a window but cannot emit a label.
// The current window must be eligible and agree with at least one of the
// previous two. Reset after a capture failure or a discontinuity.
class SpeciesConfirmation {
 public:
  SpeciesConfirmation() { reset(); }
  void reset() {
    count_=0;
    for(uint8_t i=0;i<3;++i) history_[i]=-1;
  }
  int8_t update(int8_t candidate) {
    history_[0]=history_[1];history_[1]=history_[2];history_[2]=candidate;
    if(count_<3) ++count_;
    if(count_<3 || candidate<0) return -1;
    uint8_t votes=0;
    for(uint8_t i=0;i<3;++i) if(history_[i]==candidate) ++votes;
    return votes>=2?candidate:-1;
  }
 private:
  int8_t history_[3];
  uint8_t count_;
};
} // namespace mosquito
#endif
