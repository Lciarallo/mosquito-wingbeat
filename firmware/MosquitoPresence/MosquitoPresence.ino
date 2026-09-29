/*
  Prototype presence detector for Arduino Nano 33 BLE Sense / Sense Rev2.
  Select Arduino Mbed OS Nano Boards -> Arduino Nano 33 BLE in the IDE.
  Serial output is an acoustic candidate score, not species or insect count.
  See firmware/README.md for validation results and operating-point tradeoffs.
*/
#include <PDM.h>
#include "StreamingFeatures.h"
#include "PresenceModel.h"

mosquito::StreamingFeatures frontend;
mosquito::Confirmation confirmation;
float featureVector[mosquito::kFeatureCount];
// Moderate is the shipped default. These modes have different measured
// false positive and missed-positive rates; changing it is a deliberate choice.
const float detectionThreshold = mosquito::kModerateThreshold;

const uint16_t ringSize=4096;
volatile int16_t pcmRing[ringSize];
volatile uint16_t ringHead=0,ringTail=0;
volatile uint32_t lostSamples=0;
uint32_t observedLostSamples=0,clippedSamples=0,windowSamples=0;
uint16_t largestSample=0;

void onPDMdata() {
  int16_t chunk[256];
  while(PDM.available()>0) {
    int count=PDM.available();
    if(count>static_cast<int>(sizeof(chunk))) count=sizeof(chunk);
    const int bytesRead=PDM.read(chunk,count);
    if(bytesRead<=0) break;
    count=bytesRead/sizeof(int16_t);
    for(int i=0;i<count;++i) {
      const uint16_t next=(ringHead+1)&(ringSize-1);
      if(next==ringTail) { ++lostSamples; continue; }
      pcmRing[ringHead]=chunk[i]; ringHead=next;
    }
  }
}

void setup() {
  pinMode(LED_BUILTIN,OUTPUT);
  Serial.begin(115200);
  const uint32_t start=millis();
  while(!Serial && millis()-start<3000) {}
  PDM.onReceive(onPDMdata);
  PDM.setBufferSize(512);
  PDM.setGain(20);
  if(!PDM.begin(1,mosquito::kSampleRate)) {
    Serial.println("Falha ao iniciar microfone PDM.");
    while(true) { digitalWrite(LED_BUILTIN,!digitalRead(LED_BUILTIN));delay(500); }
  }
  Serial.println("Detector experimental: 16 kHz; 0,992 s; confirmacao 2/3.");
}

void loop() {
  int16_t sample=0;bool available=false,discontinuity=false;
  noInterrupts();
  if(lostSamples!=observedLostSamples) {
    observedLostSamples=lostSamples;ringTail=ringHead;discontinuity=true;
  }
  if(ringTail!=ringHead) {
    sample=pcmRing[ringTail];ringTail=(ringTail+1)&(ringSize-1);available=true;
  }
  interrupts();
  if(discontinuity) {
    frontend.reset();confirmation.reset();windowSamples=clippedSamples=largestSample=0;
    digitalWrite(LED_BUILTIN,LOW);
    Serial.print("Audio interrompido; amostras perdidas=");Serial.println(observedLostSamples);
  }
  if(!available) return;
  const uint16_t magnitude=sample<0?-static_cast<int32_t>(sample):sample;
  if(magnitude>largestSample) largestSample=magnitude;
  if(magnitude>=32700) ++clippedSamples;
  ++windowSamples;
  if(frontend.push(sample,featureVector)) {
    // These guards only flag capture failures. Their field rates are unmeasured.
    const bool qualityOK=largestSample>=32 && clippedSamples*100<=windowSamples;
    const float score=mosquito::presenceScore(featureVector);
    bool confirmed=false;
    if(qualityOK) confirmed=confirmation.update(score>=detectionThreshold);
    else confirmation.reset();
    digitalWrite(LED_BUILTIN,confirmed?HIGH:LOW);
    Serial.print("escore=");Serial.print(score,4);
    Serial.print(" limiar=");Serial.print(detectionThreshold,4);
    Serial.print(" audio_ok=");Serial.print(qualityOK?1:0);
    Serial.print(" candidato_persistente=");Serial.println(confirmed?1:0);
    windowSamples=clippedSamples=largestSample=0;
  }
}
