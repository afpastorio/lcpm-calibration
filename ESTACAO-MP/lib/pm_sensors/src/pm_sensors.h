#ifndef pm_sensors_h
#define pm_sesnsors_h

#include <Arduino.h>

#include <iostream>

enum MODE {
    PASSIVE = 0,
    CONTINUOUS
};

enum SENSOR_STATUS {
    SLEEPING = 0,
    AWAKE
};

class PM_Sensor {
   public:
    PM_Sensor(){};
    virtual int Read() = 0;
    virtual int setMode(MODE m);
    MODE getMode();
    SENSOR_STATUS getStatus();
    virtual void Sleep();
    virtual void WakeUp();
    virtual double getPM10();
    virtual double getPM25();
    // virtual int getPM1() = 0;

   protected:
    double PM10_value, PM25_value;
    MODE _mode;
    Stream *bus;
    SENSOR_STATUS _status;
};

class PMS7003 : public PM_Sensor {
   public:
    PMS7003(HardwareSerial *hs, MODE m, unsigned long baud_rate);
    int Read();
    int setMode(MODE m);
    void Sleep();
    void WakeUp();
    int getPM1();

   private:
    double PM1_value;
};

class SDS : public PM_Sensor {
   public:
    SDS(HardwareSerial *hs, MODE m, unsigned long baud_rate);
    int Read();
    int setMode(MODE m);
    void Sleep();
    void WakeUp();
};

class HPMA115S0 : public PM_Sensor {
   public:
    HPMA115S0(HardwareSerial *hs, MODE m, unsigned long baud_rate);
    int Read();
    int setMode(MODE m);
    void Sleep();
    void WakeUp();
};

#endif