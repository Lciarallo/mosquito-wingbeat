#ifndef MOSQUITO_STREAMING_FEATURES_H
#define MOSQUITO_STREAMING_FEATURES_H
#include <math.h>
#include <stdint.h>
#include <string.h>

namespace mosquito {
static const uint16_t kSampleRate = 16000;
static const uint16_t kFrameSize = 512;
static const uint8_t kWindowFrames = 31;
static const uint8_t kBands = 31;
static const uint8_t kFeatureCount = 68;

// No heap allocation, full-second PCM buffer, or external DSP library.
class StreamingFeatures {
 public:
  StreamingFeatures() {
    for (uint16_t n=0; n<kFrameSize; ++n)
      hann_[n] = 0.5f - 0.5f*cosf(6.283185307179586f*n/(kFrameSize-1));
    reset();
  }
  void reset() {
    samples_ = frames_ = 0;
    memset(sum_, 0, sizeof(sum_)); memset(square_, 0, sizeof(square_));
  }
  // Returns true when exactly 31 consecutive complete frames are available.
  bool push(int16_t sample, float output[kFeatureCount]) {
    real_[samples_++] = static_cast<float>(sample);
    if (samples_ != kFrameSize) return false;
    samples_ = 0;
    processFrame();
    if (++frames_ != kWindowFrames) return false;
    for (uint8_t b=0; b<kBands; ++b) {
      output[b] = sum_[b];
      output[kBands+b] = deviation(b);
    }
    for (uint8_t b=kBands; b<kBands+3; ++b) {
      const uint8_t index = 2*kBands + 2*(b-kBands);
      output[index] = sum_[b];
      output[index+1] = deviation(b);
    }
    reset();
    return true;
  }
 private:
  float real_[kFrameSize], imag_[kFrameSize], hann_[kFrameSize];
  float sum_[kBands+3], square_[kBands+3];
  uint16_t samples_;
  uint8_t frames_;
  float deviation(uint8_t b) const {
    return sqrtf(fmaxf(0.f,square_[b]/kWindowFrames));
  }
  void accumulate(uint8_t b,float value) {
    const float delta=value-sum_[b];
    sum_[b]+=delta/(frames_+1);
    square_[b]+=delta*(value-sum_[b]);
  }
  void fft() {
    for (uint16_t i=1,j=0; i<kFrameSize; ++i) {
      uint16_t bit=kFrameSize>>1;
      for (; j&bit; bit>>=1) j^=bit;
      j^=bit;
      if (i<j) {
        float tmp=real_[i]; real_[i]=real_[j]; real_[j]=tmp;
        tmp=imag_[i]; imag_[i]=imag_[j]; imag_[j]=tmp;
      }
    }
    for (uint16_t length=2; length<=kFrameSize; length<<=1) {
      const float angle=-6.283185307179586f/length;
      const float cosine=cosf(angle), sine=sinf(angle);
      for (uint16_t start=0; start<kFrameSize; start+=length) {
        float wr=1.f,wi=0.f;
        for (uint16_t j=0;j<length/2;++j) {
          const uint16_t even=start+j,odd=even+length/2;
          const float tr=wr*real_[odd]-wi*imag_[odd];
          const float ti=wr*imag_[odd]+wi*real_[odd];
          const float er=real_[even],ei=imag_[even];
          real_[even]=er+tr; imag_[even]=ei+ti;
          real_[odd]=er-tr; imag_[odd]=ei-ti;
          const float next=wr*cosine-wi*sine;
          wi=wr*sine+wi*cosine; wr=next;
        }
      }
    }
  }
  void processFrame() {
    float mean=0.f;
    for(uint16_t n=0;n<kFrameSize;++n) mean+=real_[n];
    mean/=kFrameSize;
    for(uint16_t n=0;n<kFrameSize;++n) {
      real_[n]=(real_[n]-mean)*hann_[n]; imag_[n]=0.f;
    }
    fft();
    // Reuse the real array for powers after the FFT has completed.
    for(uint16_t n=4;n<128;++n) real_[n]=real_[n]*real_[n]+imag_[n]*imag_[n];
    float total=0.f;
    for(uint16_t n=4;n<128;++n) total+=real_[n];
    total=fmaxf(total,1e-20f);
    for(uint8_t b=0;b<kBands;++b) {
      float energy=0.f;
      for(uint8_t k=0;k<4;++k) energy+=real_[4+4*b+k];
      accumulate(b,log10f(fmaxf(energy/total,1e-8f)));
    }
    float logMean=0.f;
    for(uint16_t n=4;n<128;++n) logMean+=logf(fmaxf(real_[n]/total,1e-12f));
    accumulate(kBands,expf(logMean/124.f)*124.f);
    uint8_t peak=7;
    for(uint8_t n=8;n<29;++n) if(real_[n]>real_[peak]) peak=n;
    accumulate(kBands+1,real_[peak]/total);
    accumulate(kBands+2,peak*(31.25f/1000.f));
  }
};

class Confirmation {
 public:
  Confirmation(): bits_(0),count_(0) {}
  void reset() { bits_=count_=0; }
  bool update(bool detected) {
    bits_=((bits_<<1)|(detected?1:0))&7;
    if(count_<3) ++count_;
    const uint8_t votes=(bits_&1)+((bits_>>1)&1)+((bits_>>2)&1);
    return count_==3 && votes>=2;
  }
 private:
  uint8_t bits_,count_;
};
} // namespace mosquito
#endif
