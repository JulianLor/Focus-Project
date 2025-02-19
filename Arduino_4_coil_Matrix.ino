#include <math.h>

// Constants
const float Hz = 1;             // Frequency of the AC signal
const float I_max = 64.0;         // Maximum amplitude for PWM (0-255 for 8-bit PWM)
const float rot_freq = 10;        // Rotation frequency
const float phi = PI / 2;    // Example phase angle adjustment in radians (30 degrees)

const float M[4][3] = {
    {1/(2 * sin(phi)), 0, 1/(4 * cos(phi))},
    {-1/(2 * sin(phi)), 0, 1/(4 * cos(phi))},
    {0, 1/(2 * sin(phi)), 1/(4 * cos(phi))},
    {0, -1/(2 * sin(phi)), 1/(4 * cos(phi))}
    };

// H-bridge PWM pins for each coil
const int pwmPin1A = 10, pwmPinINA1 = 12, pwmPinINB1 = 9;
const int pwmPin2A = 11, pwmPinINA2 = 13, pwmPinINB2 = 8;
const int pwmPin3A = 6, pwmPinINA3 = 7, pwmPinINB3 = 3;
const int pwmPin4A = 5, pwmPinINA4 = 4, pwmPinINB4 = 2;

void setup() {
    pinMode(pwmPin1A, OUTPUT); pinMode(pwmPinINA1, OUTPUT); pinMode(pwmPinINB1, OUTPUT);
    pinMode(pwmPin2A, OUTPUT); pinMode(pwmPinINA2, OUTPUT); pinMode(pwmPinINB2, OUTPUT);
    pinMode(pwmPin3A, OUTPUT); pinMode(pwmPinINA3, OUTPUT); pinMode(pwmPinINB3, OUTPUT);
    pinMode(pwmPin4A, OUTPUT); pinMode(pwmPinINA4, OUTPUT); pinMode(pwmPinINB4, OUTPUT);

    Serial.begin(9600); // For debugging
}

void MatrixMultiplication(float vector[3], float current[4]) {
    for (int i = 0; i < 4; i++) {
        current[i] = 0;  // Initialize result row
        for (int j = 0; j < 3; j++) {
            current[i] += M[i][j] * vector[j];
          }
      }
}

void rotateVector(float vector[3], char axis, float theta, float result[3]) {
    float rotationMatrix[3][3];

    // Choose the correct rotation matrix
    if (axis == 'x') {
        float cosT = cos(theta);
        float sinT = sin(theta);
        float tempMatrix[3][3] = {
            {1, 0, 0},
            {0, cosT, -sinT},
            {0, sinT, cosT}
        };
        memcpy(rotationMatrix, tempMatrix, sizeof(tempMatrix));
    } 
    else if (axis == 'y') {
        float cosT = cos(theta);
        float sinT = sin(theta);
        float tempMatrix[3][3] = {
            {cosT, 0, sinT},
            {0, 1, 0},
            {-sinT, 0, cosT}
        };
        memcpy(rotationMatrix, tempMatrix, sizeof(tempMatrix));
    } 
    else (axis == 'z') {
        float cosT = cos(theta);
        float sinT = sin(theta);
        float tempMatrix[3][3] = {
            {cosT, -sinT, 0},
            {sinT, cosT, 0},
            {0, 0, 1}
        };
        memcpy(rotationMatrix, tempMatrix, sizeof(tempMatrix));
    } 

    // Perform the matrix-vector multiplication
    for (int i = 0; i < 3; i++) {
        result[i] = 0;
        for (int j = 0; j < 3; j++) {
            result[i] += rotationMatrix[i][j] * vector[j];
        }
    }
}

void loop() {
    float t = millis() / 1000.0; // Time in seconds
    float t_state = fmod(t, rot_freq);
    float t_theta = fmod(t_state, rot_freq / 3);
    float theta = t_theta * PI / (2 * rot_freq / 3);

    float x = sin(Hz * t * 2 * PI);
    float y = sin(Hz * t * 2 * PI);
    float z = 0;

    float direction[3] = {x, y, z};

    float vector[3];
    float current[4];

    // x-axis rot
    if (t_state < rot_freq / 3) {
        rotateVector(direction, 'x', theta, vector);
        MatrixMultiplication(vector, current);
    }

    // y-axis rot
    else if (t_state < (2 * rot_freq / 3)) {
        rotateVector(direction, 'y', theta, vector);
        MatrixMultiplication(vector, current);
    }

    // z-xis rot
    else {
        rotateVector(direction, 'z', theta, vector);
        MatrixMultiplication(vector, current);
    }

    float current1, current2, current3, current4;
    current1 = current[0];
    current2 = current[1];
    current3 = current[2];
    current4 = current[4];

    int pwmPinINA1_val, pwmPinINB1_val, pwmPinINA2_val, pwmPinINB2_val, pwmPinINA3_val, pwmPinINB3_val, pwmPinINA4_val, pwmPinINB4_val;

    // Enabler pins for Coil 1 
    if (current1 < 0) {
        pwmPinINA1_val = HIGH;
        pwmPinINB1_val = LOW;
    }
    else {
        pwmPinINA1_val = LOW;
        pwmPinINB1_val = HIGH;
    }

    // Enabler pins for Coil 2
    if (current2 < 0) {
        pwmPinINA2_val = HIGH;
        pwmPinINB2_val = LOW;
    }
    else {
        pwmPinINA2_val = LOW;
        pwmPinINB2_val = HIGH;
    }

    // Enabler pins for Coil 3
    if (current3 < 0) {
        pwmPinINA3_val = HIGH;
        pwmPinINB3_val = LOW;
    }
    else {
        pwmPinINA3_val = LOW;
        pwmPinINB3_val = HIGH;
    }

    // Enabler pins for Coil 4
    if (current4 < 0) {
        pwmPinINA4_val = HIGH;
        pwmPinINB4_val = LOW;
    }
    else {
        pwmPinINA4_val = LOW;
        pwmPinINB4_val = HIGH;
    }

    digitalWrite(pwmPinINA1, pwmPinINA1_val);
    digitalWrite(pwmPinINB1, pwmPinINB1_val);

    digitalWrite(pwmPinINA2, pwmPinINA2_val);
    digitalWrite(pwmPinINB2, pwmPinINB2_val);

    digitalWrite(pwmPinINA3, pwmPinINA3_val);
    digitalWrite(pwmPinINB3, pwmPinINB3_val);

    digitalWrite(pwmPinINA4, pwmPinINA4_val);
    digitalWrite(pwmPinINB4, pwmPinINB4_val);

    float current1_abs = abs(current1);
    float current2_abs = abs(current2);
    float current3_abs = abs(current3);
    float current4_abs = abs(current4);

    Serial.print("Current1 output:");
    Serial.println(current1);
    Serial.print("Current2 output:");
    Serial.println(current2);
    Serial.print("Current3 output:");
    Serial.println(current3);
    Serial.print("Current4 output:");
    Serial.println(current4); 

    int pwmValue1 = int(current1_abs * I_max); // Scale sine to 0-255
    int pwmValue2 = int(current2_abs * I_max); // Scale cosine to 0-255
    int pwmValue3 = int(current3_abs * I_max); // Scale sine to 0-255
    int pwmValue4 = int(current4_abs * I_max); // Scale cosine to 0-255

    analogWrite(pwmPin1A, pwmValue1);  
    analogWrite(pwmPin2A, pwmValue2);
    analogWrite(pwmPin3A, pwmValue3);  
    analogWrite(pwmPin4A, pwmValue4);
      
    delay(10);  // Short delay to manage loop frequency
  }
  
