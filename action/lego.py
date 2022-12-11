from buildhat import Motor, Matrix
import time


class Action():
    def __init__(self):
        self.motor = Motor('A')
        self.motor.set_default_speed(25)
#         self.motor.when_rotated = self._handle_motor
        self.matrix = Matrix('B')
        """
        matrix colors:
        0-''; 1-pink; 2-lilac; 3-blue; 4-cyan; 5-turquoise;
        6-green; 7-yellow; 8-orange; 9-red; 10-white
        """
        self.ok_color = [[(6, 10) for x in range(3)] for y in range(3)]
        self.nok_color = [[(9, 10) for x in range(3)] for y in range(3)]
        self.matrix.set_transition(2) #fade-in/out
        self.matrix.set_pixel((1, 1), ("blue", 10))
#         self.matrix.set_pixels(self.nok_color)

    def _handle_motor(self, speed, pos, apos):
        """Motor data
        :param speed: Speed of motor
        :param pos: Position of motor
        :param apos: Absolute position of motor
        """
        print("Motor:", speed, pos, apos)

    def run(self, action_status):
        while True:
            print(f"$$$$$$$$$$$$$$$$$$$$$ action_status[0]:{action_status[0]}")
            if action_status[0] == 'Allowed':
                self.matrix.set_pixels(self.ok_color)
                time.sleep(1)
                self.motor.run_for_degrees(-90, blocking=False)
                time.sleep(5)
                self.motor.run_for_degrees(90, blocking=False)
                time.sleep(1)
#                 self.matrix.clear()
                #run_to_position(degrees, speed=None, blocking=True, direction='shortest')  #shortest (default)/clockwise/anticlockwis  #Position in degrees from -180 to 180
                #print("Position: ", motor_a.get_aposition())         
            elif action_status[0] == 'Prohibited':
                self.matrix.set_pixels(self.nok_color)
                time.sleep(3)
#                 self.matrix.clear()
            else:
                self.matrix.clear()
                self.matrix.set_pixel((1, 1), ("blue", 10))
                time.sleep(1)
                self.matrix.set_pixel((1, 1), (0, 10))
                time.sleep(1)
                



if __name__ == "__main__":
    action = Action()
    action.run(['Allowed'])
    time.sleep(3)
    action.run([None])


