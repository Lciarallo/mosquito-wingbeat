// Native validation bridge; never included in the Arduino sketch build.
#include "MosquitoPresence/StreamingFeatures.h"
#include "MosquitoPresence/PresenceModel.h"
#include <stddef.h>
extern "C" {
int mosquito_features(const int16_t* pcm,size_t count,float* output) {
  mosquito::StreamingFeatures frontend;
  bool complete=false;
  for(size_t i=0;i<count;++i) complete=frontend.push(pcm[i],output)||complete;
  return complete?1:0;
}
float mosquito_score(const float* features) { return mosquito::presenceScore(features); }
size_t mosquito_frontend_bytes() { return sizeof(mosquito::StreamingFeatures); }
void mosquito_confirm(const int* detected,size_t count,int* output) {
  mosquito::Confirmation state;
  for(size_t i=0;i<count;++i) output[i]=state.update(detected[i]!=0)?1:0;
}
}
