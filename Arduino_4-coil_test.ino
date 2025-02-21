#include <math.h>

// Constants
const float Hz = 1;             // Frequency of the AC signal
const float I_max = 128.0;         // Maximum amplitude for PWM (0-255 for 8-bit PWM)
const float rot_freq = 1.0;        // Rotation frequency
const float angle_adj = 1.57079;    // Example phase angle adjustment in radians (30 degrees)
const float angle_opp = 1.57079;    // Example opposite angle in radians (60 degrees)

// H-bridge PWM pins for each coil
const int pwmPin1A = 9, pwmPinINA1 = 2, pwmPinINB1 = 3;
const int pwmPin2A = 11, pwmPinINA2 = 4, pwmPinINB2 = 5;
const int pwmPin3A = 10, pwmPinINA3 = 7, pwmPinINB3 = 8;
const int pwmPin4A = 6, pwmPinINA4 = 12, pwmPinINB4 = 13;

void setup() {
    // Set PWM pins as outputs
    pinMode(pwmPin1A, OUTPUT); pinMode(pwmPinINA1, OUTPUT); pinMode(pwmPinINB1, OUTPUT);
    pinMode(pwmPin2A, OUTPUT); pinMode(pwmPinINA2, OUTPUT); pinMode(pwmPinINB2, OUTPUT);
    pinMode(pwmPin3A, OUTPUT); pinMode(pwmPinINA3, OUTPUT); pinMode(pwmPinINB3, OUTPUT);
    pinMode(pwmPin4A, OUTPUT); pinMode(pwmPinINA4, OUTPUT); pinMode(pwmPinINB4, OUTPUT);

    Serial.begin(9600); // For debugging
}

void loop() {
    // Time in seconds (adjusted with millis)
    float t = millis() / 1000.0;
    
    // Calculate current values for the sinusoidal wave with a phase offset
    float sineVal = sin(2 * PI * Hz * t);
    float current2, current3, current4;

    if (t_state < rot_freq / 6) {
        current2 = sin(PI * (2 * Hz * t - 1));
        current3 = sin(PI * (2 * Hz * t) - angle_adj);
        current4 = sin(PI * (2 * Hz * t - 1) - angle_adj);
    } 
    else if (t_state < rot_freq / 3) {
        float t_state1 = t_state - (rot_freq / 6);
        float offset = t_state1 / (rot_freq / 6.0);

        current2 = sin(PI * (2 * Hz * t - 1) + offset * (PI - angle_opp));
        current3 = sin(PI * (2 * Hz * t) + angle_adj * (offset - 1));
        current4 = sin(PI * (2 * Hz * t - 1) - angle_adj + offset * (PI + angle_adj - angle_opp));
    } 
    else if (t_state < rot_freq / 2) {
        current2 = sin(PI * (2 * Hz * t) - angle_opp);
        current3 = current1;
        current4 = current2;
    } 
    else if (t_state < 2 * rot_freq / 3) {
        float t_state1 = t_state - (rot_freq / 2);
        float offset = t_state1 / (rot_freq / 6.0);

        current2 = sin(PI * (2 * Hz * t - 0.5 + offset / 60.0));
        current3 = sin(PI * (2 * Hz * t - offset / 30.0));
        current4 = sin(PI * (2 * Hz * t + offset / 20.0));
    } 
    else if (t_state < 5 * rot_freq / 6) {
        current2 = sin(PI * (2 * Hz * t) - angle_adj);
        current3 = current2;
        current4 = current1;
    } 
    else {
        float t_state1 = t_state - (5 * rot_freq / 6);
        float offset = t_state1 / (rot_freq / 6.0);

        current2 = sin(PI * (2 * Hz * t) - angle_adj + offset * (angle_adj - PI));
        current3 = sin(PI * (2 * Hz * t) - angle_adj);
        current4 = sin(PI * (2 * Hz * t) - offset * (1 + angle_adj));
    }

    digitalWrite(pwmPinINA1, sineVal < 0 ? HIGH : LOW);
    digitalWrite(pwmPinINB1, sineVal < 0 ? LOW : HIGH);

    digitalWrite(pwmPinINA2, cosineVal < 0 ? HIGH : LOW);
    digitalWrite(pwmPinINB2, cosineVal < 0 ? LOW : HIGH);

    digitalWrite(pwmPinINA3, sineValb < 0 ? HIGH : LOW);
    digitalWrite(pwmPinINB3, sineValb < 0 ? LOW : HIGH);

    digitalWrite(pwmPinINA4, cosineValb < 0 ? HIGH : LOW);
    digitalWrite(pwmPinINB4, cosineValb < 0 ? LOW : HIGH);

    float sineVal_abs = abs(sineVal);
    float cosineVal_abs = abs(cosineVal);
    float sineValb_abs = abs(sineValb);
    float cosineValb_abs = abs(cosineValb);

    int pwmValue1 = int(sineVal_abs * I_max); // Scale sine to 0-255
    int pwmValue2 = int(cosineVal_abs * I_max); // Scale cosine to 0-255
    int pwmValue3 = int(sineValb_abs * I_max); // Scale sine to 0-255
    int pwmValue4 = int(cosineValb_abs * I_max); // Scale cosine to 0-255

    analogWrite(pwmPin1A, pwmValue1);  
    analogWrite(pwmPin2A, pwmValue2);
    analogWrite(pwmPin3A, pwmValue3);  
    analogWrite(pwmPin4A, pwmValue4);
    
    delay(10);  // Short delay to manage loop frequency
}
