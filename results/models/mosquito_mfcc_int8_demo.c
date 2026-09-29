#include <stdio.h>
#include "mosquito_mfcc_int8.h"
int main(void) {
float input[MOSQUITO_FEATURES];
while (1) {
for (int j=0;j<MOSQUITO_FEATURES;++j) if (scanf("%f",&input[j])!=1) return 0;
printf("%d\n",mosquito_predict_mfcc(input));
}
}
