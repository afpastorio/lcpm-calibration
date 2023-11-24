#include "pm_sensors.h"

#include <Arduino.h>

#include <iostream>

using namespace std;

// PM_Sensor::PM_Sensor(HardwareSerial *hs, MODE m, unsigned long baud_rate): _mode(m) {
//     hs->begin(baud_rate);
//     bus = hs;
// }

MODE PM_Sensor::getMode() {
    return _mode;
}

SENSOR_STATUS PM_Sensor::getStatus() {
    return _status;
}

double PM_Sensor::getPM10() {
    return PM10_value;
}

double PM_Sensor::getPM25() {
    return PM25_value;
}

PMS7003::PMS7003(HardwareSerial *hs, MODE m, unsigned long baud_rate) {
    hs->begin(baud_rate, SERIAL_8N1);
    bus = hs;
    delay(5000);
    this->setMode(m);
    this->_status = AWAKE;
}

int PMS7003::setMode(MODE m) {
    // Serial.println("^^Passive^^");
    if (m == PASSIVE) {
        uint8_t command[] = {0x42, 0x4D, 0xE1, 0x00, 0x00, 0x01, 0x70};
        bus->write(command, sizeof(command));
        bus->flush();
        uint8_t payload[8];
        if (bus->available()) {
            bus->readBytes(payload, 8);
        }
        // for (int i = 0; i < 8; i++) {
        //     Serial.print(payload[i], HEX);
        // }
        // Serial.println("* Passive *");
    } else if (m == CONTINUOUS) {
        uint8_t command[] = {0x42, 0x4D, 0xE1, 0x00, 0x01, 0x01, 0x71};
        bus->write(command, sizeof(command));
        bus->flush();
        uint8_t payload[8];
        if (bus->available()) {
            bus->readBytes(payload, 8);
        }
        // for (int i = 0; i < 8; i++) {
        //     Serial.print(payload[i], HEX);
        // }
        // Serial.println("* Continuous *");
    } else {
        throw invalid_argument("Mode different from PASSIVE/CONTINUOUS.");
    }
    _mode = m;
    return 1;
}

void PMS7003::Sleep() {
    uint8_t command[] = {0x42, 0x4D, 0xE4, 0x00, 0x00, 0x01, 0x73};
    bus->write(command, sizeof(command));
    bus->flush();
    uint8_t payload[8];

    while (bus->available()) {
        bus->readBytes(payload, 8);
    }
    // for (int i = 0; i < 8; i++) {
    //     Serial.print(payload[i], HEX);
    // }
    // Serial.println("* Sleep *");
    _status = SLEEPING;
}

void PMS7003::WakeUp() {
    uint8_t command[] = {0x42, 0x4D, 0xE4, 0x00, 0x01, 0x01, 0x74};
    bus->write(command, sizeof(command));
    bus->flush();
    // Serial.println("* Wake *");
    _status = AWAKE;
}

int PMS7003::Read() {
    if (_status == SLEEPING) WakeUp();

    delay(5000);

    if (_mode == PASSIVE) {
        uint8_t command[] = {0x42, 0x4D, 0xE2, 0x00, 0x00, 0x01, 0x71};  // Request read
        bus->write(command, sizeof(command));
        bus->flush();
    }
    uint8_t payload[32];
    int i = 0;
    bool head = false;
    bool subhead = false;
    if (bus->available()) {
        while (bus->peek() != 0x42) bus->read();
        while (bus->available()) {
            if (i < 32)
                payload[i] = bus->read();
            if (payload[i] == 0x42)
                head = true;
            if (payload[i] == 0x4d)
                subhead = true;
            if ((head && subhead) && (bus->peek() == 0x42)) {
                i = 0;
                head = false;
                subhead = false;
            } else
                i++;
            /* code */
        }

        // bus->readBytes(payload, 32);
    }

    int payloadChecksum = payload[31] + (payload[30] << 8);
    int calculatedChecksum = 0x0;

    for (int i = 0; i < 30; i++) {
        calculatedChecksum += payload[i];
    }
    // for (int i = 0; i < 32; i++) {
    //     Serial.print(payload[i], HEX);
    // }
    // // bus->flush();
    // Serial.println("* Read *");

    if (!(payloadChecksum - calculatedChecksum)) {
        // Serial.println("* OK: Read *");
        PM1_value = payload[5] + (payload[4] << 8);
        PM25_value = payload[7] + (payload[6] << 8);
        PM10_value = payload[9] + (payload[8] << 8);
    }
    // } else
        // Serial.println("* FAIL: Read *");

    delay(5000);

    if (_mode == PASSIVE && _status == AWAKE) Sleep();

    return payloadChecksum - calculatedChecksum;
}

int PMS7003::getPM1() {
    return PM1_value;
}

/* -------------- HPMA -------------- */
HPMA115S0::HPMA115S0(HardwareSerial *hs, MODE m, unsigned long baud_rate) {
    hs->begin(baud_rate, SERIAL_8N1);
    bus = hs;
    delay(5000);
    this->setMode(m);
    this->_status = AWAKE;
}

int HPMA115S0::setMode(MODE m) {
    // Serial.println("^^Passive^^");
    if (m == PASSIVE) {
        uint8_t command[] = {0x68, 0x01, 0x20, 0x77};
        bus->write(command, sizeof(command));
        bus->flush();
        uint8_t payload[2];
        if (bus->available()) {
            while(bus->peek() != 0xa5) bus->read();
            bus->readBytes(payload, 2);
        }
        // for (int i = 0; i < 2; i++) {
        //     Serial.print(payload[i], HEX);
        // }
        // Serial.println("* Passive *");
    } else if (m == CONTINUOUS) {
        uint8_t command[] = {0x68, 0x01, 0x40, 0x57};
        bus->write(command, sizeof(command));
        bus->flush();
        uint8_t payload[2];
        if (bus->available()) {
            while(bus->peek() != 0xa5) bus->read();
            bus->readBytes(payload, 2);
        }
        // for (int i = 0; i < 2; i++) {
        //     Serial.print(payload[i], HEX);
        // }
        // Serial.println("* Continuous *");
    } else {
        throw invalid_argument("Mode different from PASSIVE/CONTINUOUS.");
    }
    _mode = m;
    return 1;
}

void HPMA115S0::Sleep() {
    uint8_t command[] = {0x68, 0x01, 0x02, 0x95};
    bus->write(command, sizeof(command));
    bus->flush();
    uint8_t payload[2];

    while (bus->available()) {
        bus->readBytes(payload, 2);
    }
    // for (int i = 0; i < 2; i++) {
    //     Serial.print(payload[i], HEX);
    // }
    // Serial.println("* Sleep *");
    _status = SLEEPING;
}

void HPMA115S0::WakeUp() {
    uint8_t command[] = {0x68, 0x01, 0x01, 0x96};
    bus->write(command, sizeof(command));
    bus->flush();
    uint8_t payload[2];

    while (bus->available()) {
        bus->readBytes(payload, 2);
    }
    // for (int i = 0; i < 2; i++) {
    //     Serial.print(payload[i], HEX);
    // }
    // Serial.println("* Wake *");
    _status = AWAKE;
}

int HPMA115S0::Read() {
    if (_status == SLEEPING) WakeUp();

    delay(5000);
    int error_code = 0;
    if (_mode == PASSIVE) {
        uint8_t command[] = {0x68, 0x01, 0x04, 0x93};  // Request read
        bus->write(command, sizeof(command));
        bus->flush();

        uint8_t payload[8];
        int i = 0;
        bool head = false;
        bool subhead = false;

        if (bus->available()) {
            while (bus->peek() == 0xa5) bus->read();
            bus->readBytes(payload, 8);
            // while (bus->peek() != 0x40) bus->read();
            // while (bus->available()) {
            //     if (i < 8)
            //         payload[i] = bus->read();
            //     if (payload[i] == 0x40)
            //         head = true;
            //     if (payload[i] == 0x05)
            //         subhead = true;
            //     if ((head && subhead) && (bus->peek() == 0x40)) {
            //         i = 0;
            //         head = false;
            //         subhead = false;
            //     } else
            //         i++;
            // }
        }

        int payloadChecksum = payload[7];
        int calculatedChecksum = 0x0;

        for (int i = 0; i < 7; i++) {
            calculatedChecksum += payload[i];
        }
        calculatedChecksum = (65536-calculatedChecksum)%256;
        // for (int i = 0; i < 8; i++) {
        //     Serial.print(payload[i], HEX);
        // }
        // // bus->flush();
        // Serial.println("* Read *");
        error_code = payloadChecksum - calculatedChecksum;

        if (!(payloadChecksum - calculatedChecksum)) {
            // Serial.println("* OK: Read *");
            PM25_value = payload[4] + (payload[3] << 8);
            PM10_value = payload[6] + (payload[5] << 8);
        }
        else
        {
            PM25_value = -1;
            PM10_value = -1;
        }
        // } else
            // Serial.println("* FAIL: Read *");
    }
    else if(_mode == CONTINUOUS) {
        uint8_t payload[32];
        int i = 0;
        bool head = false;
        bool subhead = false;
        if (bus->available()) {
            while (bus->peek() != 0x42) bus->read();
            while (bus->available()) {
                if (i < 32)
                    payload[i] = bus->read();
                if (payload[i] == 0x42)
                    head = true;
                if (payload[i] == 0x4d)
                    subhead = true;
                if ((head && subhead) && (bus->peek() == 0x42)) {
                    i = 0;
                    head = false;
                    subhead = false;
                } else
                    i++;
            }
        }

        int payloadChecksum = payload[31] + (payload[30] << 8);
        int calculatedChecksum = 0x0;

        for (int i = 0; i < 30; i++) {
            calculatedChecksum += payload[i];
        }
        // for (int i = 0; i < 32; i++) {
        //     Serial.print(payload[i], HEX);
        // }
        // // bus->flush();
        // Serial.println("* Read *");
        error_code = payloadChecksum - calculatedChecksum;

        if (!(payloadChecksum - calculatedChecksum)) {
            // Serial.println("* OK: Read *");
            PM25_value = payload[5] + (payload[4] << 8);
            PM10_value = payload[7] + (payload[6] << 8);
        }
        // } else
            // Serial.println("* FAIL: Read *");
    }

    delay(5000);

    if (_mode == PASSIVE && _status == AWAKE) Sleep();

    return error_code;
}

/* -------------- HPMA -------------- */
SDS::SDS(HardwareSerial *hs, MODE m, unsigned long baud_rate) {
    hs->begin(baud_rate, SERIAL_8N1);
    bus = hs;
    delay(5000);
    this->setMode(m);
    this->_status = AWAKE;
}
int SDS::Read() {
    if (_status == SLEEPING) WakeUp();

    delay(5000);
    int error_code = 0;
    if (_mode == PASSIVE) {
        uint8_t command[] = {0xAA, 0xB4, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0x02, 0xAB};  // Request read
        bus->write(command, sizeof(command));
        bus->flush();
    }
    uint8_t payload[10];
    int i = 0;
    bool head = false;
    bool subhead = false;

    if (bus->available()) {
        while (bus->peek() == 0xaa) bus->read();
        bus->readBytes(payload, 10);
        // while (bus->peek() != 0x40) bus->read();
        // while (bus->available()) {
        //     if (i < 8)
        //         payload[i] = bus->read();
        //     if (payload[i] == 0x40)
        //         head = true;
        //     if (payload[i] == 0x05)
        //         subhead = true;
        //     if ((head && subhead) && (bus->peek() == 0x40)) {
        //         i = 0;
        //         head = false;
        //         subhead = false;
        //     } else
        //         i++;
        // }
    }

    int payloadChecksum = payload[8];
    int calculatedChecksum = 0x0;

    for (int i = 2; i < 8; i++) {
        calculatedChecksum += payload[i];
    }
    calculatedChecksum = calculatedChecksum%256;
    // for (int i = 0; i < 8; i++) {
    //     Serial.print(payload[i], HEX);
    // }
    // // bus->flush();
    // Serial.println("* Read *");
    error_code = payloadChecksum - calculatedChecksum;

    if (!(payloadChecksum - calculatedChecksum)) {
        // Serial.println("* OK: Read *");
        PM25_value = (payload[2] + (payload[3] << 8))/10;
        PM10_value = (payload[4] + (payload[5] << 8))/10;
    }
    // } else
        // Serial.println("* FAIL: Read *");

    delay(5000);

    if (_mode == PASSIVE && _status == AWAKE) Sleep();

    return error_code;
}
int SDS::setMode(MODE m) {
    // Serial.println("^^Passive^^"); 
    if (m == PASSIVE) {
        uint8_t command[] = {0xAA, 0xB4, 0x02, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0x02, 0xAB};
        bus->write(command, sizeof(command));
        bus->flush();
        uint8_t payload[10];
        if (bus->available()) {
            while(bus->peek() != 0xaa) bus->read();
            bus->readBytes(payload, 10);
        }
        // for (int i = 0; i < 2; i++) {
        //     Serial.print(payload[i], HEX);
        // }
        // Serial.println("* Passive *");
    } else if (m == CONTINUOUS) {
        uint8_t command[] = {0xAA, 0xB4, 0x02, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0x01, 0xAB};
        bus->write(command, sizeof(command));
        bus->flush();
        uint8_t payload[10];
        if (bus->available()) {
            while(bus->peek() != 0xaa) bus->read();
            bus->readBytes(payload, 10);
        }
        // for (int i = 0; i < 2; i++) {
        //     Serial.print(payload[i], HEX);
        // }
        // Serial.println("* Continuous *");
    } else {
        throw invalid_argument("Mode different from PASSIVE/CONTINUOUS.");
    }
    _mode = m;
    return 1;
}
void SDS::Sleep() {
    uint8_t command[] = {0xAA, 0xB4, 0x06, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0x05, 0xAB};
    bus->write(command, sizeof(command));
    bus->flush();
    uint8_t payload[10];

    while (bus->available()) {
        bus->readBytes(payload, 10);
    }
    // for (int i = 0; i < 2; i++) {
    //     Serial.print(payload[i], HEX);
    // }
    // Serial.println("* Sleep *");
    _status = SLEEPING;
}
void SDS::WakeUp() {
    uint8_t command[] = {0xAA, 0xB4, 0x06, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0x06, 0xAB};
    bus->write(command, sizeof(command));
    bus->flush();
    uint8_t payload[10];

    while (bus->available()) {
        bus->readBytes(payload, 10);
    }
    // for (int i = 0; i < 2; i++) {
    //     Serial.print(payload[i], HEX);
    // }
    // Serial.println("* Wake *");
    _status = AWAKE;
}