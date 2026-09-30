#include "MosquitoSpecies/StreamingFeatures.h"
#include "MosquitoSpecies/PresenceModel.h"
#include "MosquitoSpecies/SpeciesModel.h"
#include "MosquitoSpecies/SpeciesDecision.h"
#include <stddef.h>

extern "C" {
int mosquito_species_features(const int16_t* pcm,size_t count,float* features) {
  mosquito::StreamingFeatures frontend;
  int produced=0;
  for(size_t i=0;i<count;++i) if(frontend.push(pcm[i],features)) ++produced;
  return produced;
}
void mosquito_species_batch(const float* x,size_t n,float* probabilities,float* presence,int32_t* candidates) {
  for(size_t row=0;row<n;++row) {
    const float* features=x+row*mosquito::kFeatureCount;
    float* p=probabilities+row*mosquito::kSpeciesCount;
    mosquito::speciesProbabilities(features,p);
    const int8_t best=mosquito::bestSpecies(p);
    presence[row]=mosquito::presenceScore(features);
    candidates[row]=presence[row]>=mosquito::kModerateThreshold &&
                    p[best]>=mosquito::kSpeciesConfidenceThreshold?best:-1;
  }
}
void mosquito_species_confirm(const int32_t* labels,const int32_t* reset,size_t n,int32_t* output) {
  mosquito::SpeciesConfirmation state;
  for(size_t i=0;i<n;++i) {
    if(reset[i]) state.reset();
    output[i]=state.update(static_cast<int8_t>(labels[i]));
  }
}
size_t mosquito_species_frontend_bytes() { return sizeof(mosquito::StreamingFeatures); }
int mosquito_species_fold() { return mosquito::kSpeciesFold; }
}
