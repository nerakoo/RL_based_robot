from controller import Robot, Receiver, Emitter
import sys, struct, math
import numpy as np
import mlp as ntw


class Controller:
    def __init__(self, robot):
        # Robot Parameters
        # Please, do not change these parameters
        self.robot = robot
        self.time_step = 32  # ms
        self.max_speed = 1  # m/s
       
        self.fitness = 0
        self.fitness_plus = 0
       
        self.spinning_time = 0  # 用于跟踪自旋时间的变量
        self.spinning_threshold = 10  # 自旋时间阈值，单位为时间步长   
       
        self.markers_detected = []
        # self.spinning_penalty = 1   
        self.counter = 0   
        ######################################
        ######################################
        #   检测方块颜色
        ######################################
        ######################################
        self.number_input_layer = 11
        self.detected_purple = False
        self.detected_wood = False
        self.detected_red = False
    
        self.cumulative_spinning_penalty = 0 # 初始化累积自旋惩罚
        # MLP Parameters and Variables

        ###########
        ### DEFINE below the architecture of your MLP network:

        ### Add the number of neurons for input layer, hidden layer and output layer.
        ### The number of neurons should be in between of 1 to 20.
        ### Number of hidden layers should be one or two.

        ######################################
        ######################################
        #   在这里修改MLP层
        ######################################
        ######################################
        self.number_input_layer = 11
        # Example with one hidden layers: self.number_hidden_layer = [5]
        # Example with two hidden layers: self.number_hidden_layer = [7,5]
        self.number_hidden_layer = [10, 10]
        self.number_output_layer = 2

        # Create a list with the number of neurons per layer
        self.number_neuros_per_layer = []
        self.number_neuros_per_layer.append(self.number_input_layer)
        self.number_neuros_per_layer.extend(self.number_hidden_layer)
        self.number_neuros_per_layer.append(self.number_output_layer)

        # Initialize the network
        self.network = ntw.MLP(self.number_neuros_per_layer)
        self.inputs = []

        # Calculate the number of weights of your MLP
        self.number_weights = 0
        for n in range(1, len(self.number_neuros_per_layer)):
            if (n == 1):
                # Input + bias
                self.number_weights += (self.number_neuros_per_layer[n - 1] + 1) * self.number_neuros_per_layer[n]
            else:
                self.number_weights += self.number_neuros_per_layer[n - 1] * self.number_neuros_per_layer[n]

        # Enable Motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.velocity_left = 0
        self.velocity_right = 0

        # Enable Proximity Sensors
        self.proximity_sensors = []
        for i in range(8):
            sensor_name = 'ps' + str(i)
            self.proximity_sensors.append(self.robot.getDevice(sensor_name))
            self.proximity_sensors[i].enable(self.time_step)

        # Enable Ground Sensors
        self.left_ir = self.robot.getDevice('gs0')
        self.left_ir.enable(self.time_step)
        self.center_ir = self.robot.getDevice('gs1')
        self.center_ir.enable(self.time_step)
        self.right_ir = self.robot.getDevice('gs2')
        self.right_ir.enable(self.time_step)

        # Enable Emitter and Receiver (to communicate with the Supervisor)
        self.emitter = self.robot.getDevice("emitter")
        self.receiver = self.robot.getDevice("receiver")
        self.receiver.enable(self.time_step)
        self.receivedData = ""
        self.receivedDataPrevious = ""
        self.flagMessage = False

        # Fitness value (initialization fitness parameters once)
        self.fitness_values = []
        self.fitness = 0


    def check_for_new_genes(self):
        if (self.flagMessage == True):
            # Split the list based on the number of layers of your network
            part = []
            for n in range(1, len(self.number_neuros_per_layer)):
                if (n == 1):
                    part.append((self.number_neuros_per_layer[n - 1] + 1) * (self.number_neuros_per_layer[n]))
                else:
                    part.append(self.number_neuros_per_layer[n - 1] * self.number_neuros_per_layer[n])

            # Set the weights of the network
            data = []
            weightsPart = []
            sum = 0
            for n in range(1, len(self.number_neuros_per_layer)):
                if (n == 1):
                    weightsPart.append(self.receivedData[n - 1:part[n - 1]])
                elif (n == (len(self.number_neuros_per_layer) - 1)):
                    weightsPart.append(self.receivedData[sum:])
                else:
                    weightsPart.append(self.receivedData[sum:sum + part[n - 1]])
                sum += part[n - 1]
            for n in range(1, len(self.number_neuros_per_layer)):
                if (n == 1):
                    weightsPart[n - 1] = weightsPart[n - 1].reshape(
                        [self.number_neuros_per_layer[n - 1] + 1, self.number_neuros_per_layer[n]])
                else:
                    weightsPart[n - 1] = weightsPart[n - 1].reshape(
                        [self.number_neuros_per_layer[n - 1], self.number_neuros_per_layer[n]])
                data.append(weightsPart[n - 1])
            self.network.weights = data

            # Reset fitness list
            self.fitness_values = []

    def clip_value(self, value, min_max):
        if (value > min_max):
            return min_max;
        elif (value < -min_max):
            return -min_max;
        return value;

    def sense_compute_and_actuate(self):
        # MLP:
        #   Input == sensory data
        #   Output == motors commands
        output = self.network.propagate_forward(self.inputs)
        self.velocity_left = output[0]
        self.velocity_right = output[1]

        # Multiply the motor values by 3 to increase the velocities
        self.left_motor.setVelocity(self.velocity_left * 3)
        self.right_motor.setVelocity(self.velocity_right * 3)

    ######################################
    ######################################
    #   在这里修改fitness函数
    ######################################
    ######################################
    
    def evaluate_line_tracking(self):
        # 走在线上，给予奖励    
        left_value = self.left_ir.getValue() / 1000
        center_value = self.center_ir.getValue() / 1000
        right_value = self.right_ir.getValue() / 1000
        # #
                
        lower = 0.29
        higher = 0.35
        score_point = 0
        
        if (lower <= left_value <= higher):
             score_point = score_point + 1
        if (lower <= center_value <= higher):
             score_point = score_point + 1
        if (lower <= right_value <= higher):
             score_point = score_point + 1
        
        normalized_score = score_point / 3 # 分数范围0-1
        
        return normalized_score
        
    def evaluate_outline_tracking(self):
        # 走在地面，给予惩罚    
        left_value = self.left_ir.getValue() / 1000
        center_value = self.center_ir.getValue() / 1000
        right_value = self.right_ir.getValue() / 1000
        # #
        
        lower = 0.7
        higher = 0.8
        score_point = 0
        
        if (lower <= left_value <= higher):
             score_point = score_point + 1
        if (lower <= center_value <= higher):
             score_point = score_point + 1
        if (lower <= right_value <= higher):
             score_point = score_point + 1
        
        if (score_point>=2):
            normalized_score = -6 * score_point / 3 # 分数范围0-1
        else:
            normalized_score = 0
        # return normalized_score
        return 0

    def evaluate_stable_movement(self):
        # 计算左右轮速度的平均值
        max_speed = max(self.velocity_left,self.velocity_right) + 0.000001
        
        average_speed = (abs(self.velocity_left) + abs(self.velocity_right)) / 2
        
        # 避免自旋转：左右轮速度差异的惩罚
        speed_difference = (abs(self.velocity_left - self.velocity_right)) / max_speed
        
        # print("average_speed",average_speed,speed_difference)
        if ( 0.2<= speed_difference <=0.6 ):
            normalized_score = 2 * average_speed
        elif ( 0<= speed_difference <=0.2 ):
            normalized_score = 3 * average_speed
        else:
            normalized_score = 0
        
        return normalized_score  # 确保分数不为负
    
    def bonus_value(self):
        # 走到绿点，给予奖励    
        left_value = self.left_ir.getValue() / 1000
        center_value = self.center_ir.getValue() / 1000
        right_value = self.right_ir.getValue() / 1000
        # #
        
        # print("left_value",left_value,center_value,right_value)
        
        lower = 0.48
        higher = 0.52
        score_point = 0
        temp = 0
        get = 0
        
        for i in range(8):
            temp = max(temp,self.proximity_sensors[i].getValue() / 2000)
        
        if ( 0.06<=temp ):
            get = 1
        
        if (lower <= left_value <= higher):
             score_point = score_point + 1
        if (lower <= center_value <= higher):
             score_point = score_point + 1
        if (lower <= right_value <= higher):
             score_point = score_point + 1
        # print("score_point",score_point)
        if (score_point>=3):
            score_point = 90
        else:
            score_point = 0
        
        normalized_score = get * score_point / 3 # 分数范围0-1
        
        return normalized_score
    
    def keep_away_bonus(self):
        
        score_point = 0
        # 遇到障碍时候在白地有奖励
        left_value = self.left_ir.getValue() / 1000
        center_value = self.center_ir.getValue() / 1000
        right_value = self.right_ir.getValue() / 1000
        
        for i in range(8):
            temp = self.proximity_sensors[i].getValue() / 2000
            # print("temp:",temp)
            if ( 0.06<=temp ):
                score_point = score_point - 50
                self.knowtime = self.knowtime + 1
            # if ( 0.08<=temp<=0.13 ):
                # score_point = score_point - 50
            # elif ( temp>0.13 ):
                # score_point = score_point - 50
        
        # print("score_point",score_point)        
        normalized_score = score_point 
        
        return normalized_score
    
    def calculate_fitness(self):
        line_tracking_score = self.evaluate_line_tracking()
        outline_tracking_score = self.evaluate_outline_tracking()
        stable_movement_score = self.evaluate_stable_movement()
        bonus_value = self.bonus_value()
        keep_away_bonus = self.keep_away_bonus()
        # keep_away_bonus = 0
        # print(line_tracking_score,outline_tracking_score)
        # 综合得分
        left_value = self.left_ir.getValue() / 1000
        center_value = self.center_ir.getValue() / 1000
        right_value = self.right_ir.getValue() / 1000
        
        self.counter = self.counter + 1
        
        # if self.counter%30 == 0:
            # print("now is the ",self.counter/30,"s")
            # print("left_value:",left_value,"center_value:",center_value,"right_value:",right_value)
            # for i in range(8):
                # temp = self.proximity_sensors[i].getValue()
                # print("proximity_sensors",i,"is",temp)
            # print("score:",line_tracking_score,outline_tracking_score,stable_movement_score,bonus_value,keep_away_bonus)
       
        combined_fitness = keep_away_bonus * 0.05 + (line_tracking_score + outline_tracking_score) * stable_movement_score
        # print("combined_fitness",combined_fitness)
        # 异常数值
        if combined_fitness>6:
            combined_fitness = 0
        
        # print(combined_fitness)
        # print(bonus_value)
        
        combined_fitness = combined_fitness + bonus_value * 10  
        # if keep_away_bonus == -100:
            # combined_fitness = -5
        
        # print(combined_fitness)
        
        self.fitness_values.append(combined_fitness)
        self.fitness = np.mean(self.fitness_values)
        
        
    ######################################    
        
            

    def handle_emitter(self):
        # Send the self.fitness value to the supervisor
        data = str(self.number_weights)
        data = "weights: " + data
        string_message = str(data)
        string_message = string_message.encode("utf-8")
        # print("Robot send:", string_message)
        self.emitter.send(string_message)

        # Send the self.fitness value to the supervisor
        data = str(self.fitness)
        data = "fitness: " + data
        string_message = str(data)
        string_message = string_message.encode("utf-8")
        # print("Robot send fitness:", string_message)
        self.emitter.send(string_message)

    def handle_receiver(self):
        if self.receiver.getQueueLength() > 0:
            while (self.receiver.getQueueLength() > 0):
                # Adjust the Data to our model
                # Webots 2022:
                # self.receivedData = self.receiver.getData().decode("utf-8")
                # Webots 2023:
                self.receivedData = self.receiver.getString()
                self.receivedData = self.receivedData[1:-1]
                self.receivedData = self.receivedData.split()
                x = np.array(self.receivedData)
                self.receivedData = x.astype(float)
                # print("Controller handle receiver data:", self.receivedData)
                self.receiver.nextPacket()

            # Is it a new Genotype?
            if (np.array_equal(self.receivedDataPrevious, self.receivedData) == False):
                self.flagMessage = True

            else:
                self.flagMessage = False

            self.receivedDataPrevious = self.receivedData
        else:
            # print("Controller receiver q is empty")
            self.flagMessage = False

    def run_robot(self):
    
        
    
        # Main Loop
        while self.robot.step(self.time_step) != -1:
            # This is used to store the current input data from the sensors
            self.inputs = []
            
            self.fitness_save = self.fitness
            
            self.fitness_plus = 0
            self.knowtime = 0
            self.markers_detected = []
            
            # Emitter and Receiver
            # Check if there are messages to be sent or read to/from our Supervisor
            self.handle_emitter()
            self.handle_receiver()

            # Read Ground Sensors
            left = self.left_ir.getValue()
            center = self.center_ir.getValue()
            right = self.right_ir.getValue()
            # print("Ground Sensors \n    left {} center {} right {}".format(left,center,right))

            ### Please adjust the ground sensors values to facilitate learning
            min_gs = 0
            max_gs = 1023

            if (left > max_gs): left = max_gs
            if (center > max_gs): center = max_gs
            if (right > max_gs): right = max_gs
            if (left < min_gs): left = min_gs
            if (center < min_gs): center = min_gs
            if (right < min_gs): right = min_gs

            # Normalize the values between 0 and 1 and save data
            self.inputs.append((left - min_gs) / (max_gs - min_gs))
            self.inputs.append((center - min_gs) / (max_gs - min_gs))
            self.inputs.append((right - min_gs) / (max_gs - min_gs))
            # print("Ground Sensors \n    left {} center {} right {}".format(self.inputs[0],self.inputs[1],self.inputs[2]))

            # Read Distance Sensors
            for i in range(8):
                ### Select the distance sensors that you will use
                if (i == 0 or i == 1 or i == 2 or i == 3 or i == 4 or i == 5 or i == 6 or i == 7):
                    temp = self.proximity_sensors[i].getValue()

                    ### Please adjust the distance sensors values to facilitate learning
                    min_ds = 0
                    max_ds = 4095

                    if (temp > max_ds): temp = max_ds
                    if (temp < min_ds): temp = min_ds

                    # Normalize the values between 0 and 1 and save data
                    self.inputs.append((temp - min_ds) / (max_ds - min_ds))
                    # print("Distance Sensors - Index: {}  Value: {}".format(i,self.proximity_sensors[i].getValue()))

            # GA Iteration
            # Verify if there is a new genotype to be used that was sent from Supervisor
            self.check_for_new_genes()
            # The robot's actuation (motor values) based on the output of the MLP
            self.sense_compute_and_actuate()
            # Calculate the fitnes value of the current iteration
            self.calculate_fitness()
            
            # self.fitness = self.fitness - self.fitness_save
            # End of the iteration
    

if __name__ == "__main__":
    # Call Robot function to initialize the robot
    my_robot = Robot()
    # Initialize the parameters of the controller by sending my_robot
    controller = Controller(my_robot)
    # Run the controller
    controller.run_robot()

