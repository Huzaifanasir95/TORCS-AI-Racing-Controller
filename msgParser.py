'''
Created on Apr 5, 2012

@author: lanquarden
'''

class MsgParser(object):
    '''
    A parser for received UDP messages and building UDP messages
    '''
    
    def __init__(self):
        '''Constructor'''
        self.string = ""
        self.index = 0

    def parse(self, str_sensors):
        '''Return a dictionary with tags and values from the UDP message'''
        # Convert to string if bytes
        if isinstance(str_sensors, bytes):
            str_sensors = str_sensors.decode()
        
        self.string = str_sensors
        self.index = 0
        
        sensors = {}
        
        while self.index < len(self.string):
            # Find the next opening parenthesis
            b_open = self.string.find('(', self.index)
            if b_open == -1:
                break
            
            # Find the next closing parenthesis
            b_close = self.string.find(')', b_open)
            if b_close == -1:
                break
            
            # Extract the sensor name and values
            sensor_str = self.string[b_open+1:b_close]
            sensor_name = sensor_str.split()[0]
            sensor_values = sensor_str.split()[1:]
            
            # Convert values to appropriate type
            if len(sensor_values) == 1:
                try:
                    sensor_values = [float(sensor_values[0])]
                except ValueError:
                    sensor_values = [sensor_values[0]]
            else:
                try:
                    sensor_values = [float(v) for v in sensor_values]
                except ValueError:
                    pass
            
            sensors[sensor_name] = sensor_values
            self.index = b_close + 1
        
        return sensors

    def stringify(self, sensors):
        '''Build a UDP message from a dictionary'''
        string = ""
        for sensor_name, sensor_values in sensors.items():
            string += f"({sensor_name}"
            for value in sensor_values:
                string += f" {value}"
            string += ")"
        return string
