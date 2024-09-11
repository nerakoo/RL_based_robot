第一代：

训练出可以很好实现巡线功能的个体。

```
    def evaluate_line_tracking(self):
        # 假设地面传感器的读数已经储存在self.inputs中
        ground_sensor_readings = self.inputs[-3:]  # 假设最后3个输入是地面传感器
        line_tracking_score = sum(ground_sensor_readings) / len(ground_sensor_readings)  # 平均值
        normalized_score = line_tracking_score / 4095  # 假设最大值为4095
        return normalized_score

    def evaluate_stable_movement(self):
        # 计算左右轮速度的平均值
        average_speed = (abs(self.velocity_left) + abs(self.velocity_right)) / 2
        # 避免自旋转：左右轮速度差异的惩罚
        speed_difference = abs(self.velocity_left - self.velocity_right)
        spinning_penalty = speed_difference / (2 * self.max_speed)  # 归一化到0到1的范围
        # 综合得分
        stable_movement_score = average_speed / self.max_speed - spinning_penalty
        return max(0, stable_movement_score)  # 确保分数不为负

    def calculate_fitness(self):
        line_tracking_score = self.evaluate_line_tracking()
        stable_movement_score = self.evaluate_stable_movement()

        # 综合得分
        combined_fitness = line_tracking_score + stable_movement_score
        self.fitness_values.append(combined_fitness)
        self.fitness = np.mean(self.fitness_values)
```



第二代：

```
def evaluate_line_tracking(self):
        # 假设地面传感器的读数已经储存在self.inputs中
        ground_sensor_readings = self.inputs[-3:]  # 最后3个输入是地面传感器
        line_tracking_score = sum(ground_sensor_readings) / len(ground_sensor_readings)
        normalized_score = line_tracking_score / 4095  # 假设最大值为4095
        return normalized_score

    def evaluate_stable_movement(self):
        # 计算左右轮速度的平均值
        average_speed = (abs(self.velocity_left) + abs(self.velocity_right)) / 2
        # 避免原地打转：左右轮速度差异的惩罚
        speed_difference = abs(self.velocity_left - self.velocity_right)
        spinning_penalty = speed_difference / (2 * self.max_speed)  # 归一化到0到1的范围
        # 避免惩罚变成负数
        stable_movement_score = max(0, average_speed / self.max_speed - spinning_penalty)
        return stable_movement_score

    def evaluate_collision(self):
        # 检测是否接近障碍物，使用归一化后的传感器读数
        collision_threshold = 0.4  # 您可以根据需要调整这个阈值
        is_close_to_obstacle = np.max(self.inputs[3:11]) > collision_threshold
        # 如果靠近障碍物，则引入惩罚
        collision_penalty = -1 if is_close_to_obstacle else 0
        return collision_penalty

    # def update_fitness_for_marker(self):
    #     current_sensor_values = [self.left_ir.getValue(), self.center_ir.getValue(), self.right_ir.getValue()]
    #
    #     # 定义 mark4 的范围
    #     mark4_range = (390, 410)
    #
    #     # 计算在范围内的传感器数量
    #     sensors_in_range = sum(mark4_range[0] <= sensor_value <= mark4_range[1] for sensor_value in current_sensor_values)
    #
    #     # 如果至少有两个传感器在范围内
    #     if sensors_in_range >= 2:
    #         # 降低适应度
    #         self.fitness_plus -= 0.5  # 可根据需要调整这个惩罚值
    #         print(f"检测到深蓝色区域, 总适应度: {self.fitness}")
    #     return self.fitness_plus
        
    def update_fitness_for_marker(self):
        # 当前传感器的值
        current_sensor_values = [self.left_ir.getValue(), self.center_ir.getValue(), self.right_ir.getValue()]
        
        
        # left_value = self.left_ir.getValue()
        # center_value = self.center_ir.getValue()
        # right_value = self.right_ir.getValue()
        #
        #
        # if self.counter % 32 == 0:
        #     print("Ground sensor readings:", left_value, center_value, right_value)
        #     self.counter = 0  # 重置计数器
        #
        #
        # self.counter += 1
        
        # 定义每种颜色的范围
        color_ranges = {
            "mark1": (860, 890),
            "mark2": (1290, 1300),
            "mark3": (1470, 1520),
            "mark4": (390, 410)
        }
        #
        # 定义有序的标记列表
        ordered_markers = ["mark1", "mark2", "mark3", "mark4"]

        # 按顺序检测标记
        for marker_name in ordered_markers:
            lower_bound, upper_bound = color_ranges[marker_name]
            sensors_in_range = sum(lower_bound <= sensor_value <= upper_bound for sensor_value in current_sensor_values)

            # 如果至少有两个传感器在范围内，并且之前没有检测到这个标记
            if sensors_in_range >= 2 : # and marker_name not in self.markers_detected
                self.fitness_plus -= 8  # 每个检测到的标记增加2分
                self.markers_detected.append(marker_name)
                print(f"检测到 {marker_name}, 总适应度: {self.fitness}")
                break  # 检测到新标记后立即跳出循环
                    
        return self.fitness_plus

    def calculate_fitness(self):
        line_tracking_score = self.evaluate_line_tracking()
        stable_movement_score = self.evaluate_stable_movement()
        collision_penalty = self.evaluate_collision()
    
        # 更新适应度以包括标记检测
        self.update_fitness_for_marker()
    
        # 通过乘积结合巡线得分和稳定运动得分，再加上碰撞惩罚
        combined_fitness = line_tracking_score * stable_movement_score * 100 + collision_penalty + self.update_fitness_for_marker()
    
        # 将综合得分添加到适应度值列表中
        self.fitness_values.append(combined_fitness)
    
        # 计算平均适应度，并转换为分数
        self.fitness = np.mean(self.fitness_values)
    
        return self.fitness

        
        # if self.fitness<1:
            # self.fitness = 0
        
        # return self.fitness

        
    # def calculate_fitness(self):
        # line_tracking_score = self.evaluate_line_tracking()
        # stable_movement_score = self.evaluate_stable_movement()
        
        # combined_score = line_tracking_score + stable_movement_score
        # 调整总体得分，使得接近1的得分更高
        # 这里使用了一个指数函数，当得分接近1时，其值会接近1，而偏离1时，其值会远离1
        # adjusted_score = 1 - abs(1 - combined_score)**2
    
        # self.fitness_values.append(adjusted_score)
        # self.fitness = np.mean(self.fitness_values)
    
        # 转换成10分制
        # fitness_score = 10 * self.fitness
        # return fitness_score
    
    


    # def calculate_fitness(self):
    #
    #     ###########
    #     ### DEFINE the fitness function to increase the speed of the robot and
    #     ### to encourage the robot to move forward
    #     forwardFitness = np.max(self.is_moving())
    #
    #     ###########
    #     ### DEFINE the fitness function equation to avoid collision
    #     avoidCollisionFitness = np.max(self.inputs[3:11])
    #
    #     ###########
    #     ### DEFINE the fitness function equation to avoid spining behaviour
    #     spinningFitness = 1/self.is_spinning()
    #
    #     ###########
    #     ### DEFINE the fitness function equation of this iteration which should be a combination of the previous functions
    #     combinedFitness = avoidCollisionFitness + forwardFitness + spinningFitness
    #
    #     self.fitness_values.append(combinedFitness)
    #     self.fitness = np.mean(self.fitness_values)
```

第三版

```
    ######################################
    ######################################
    #   在这里修改fitness函数
    ######################################
    ######################################
    
    def calculate_fitness(self):
        # 巡线得分
        line_tracking_score = self.evaluate_line_tracking()
    
        # 自旋惩罚
        spinning_penalty = self.evaluate_spinning_penalty()
    
        # 速度得分
        speed_score = self.evaluate_velocity_score()
    
        # 碰撞惩罚
        collision_penalty = self.evaluate_collision_penalty()
    
        # 线外扣分
        out_of_line_penalty = self.evaluate_out_of_line_penalty()
    
        # 计算最终适应度：乘积形式结合各个得分，并确保适应度不为负值
        combined_fitness = line_tracking_score * max(1 - spinning_penalty, 0) * speed_score * max(1 - collision_penalty, 0) * max(1 - out_of_line_penalty, 0)
        self.fitness_values.append(combined_fitness)
    
        # 计算平均适应度，并转换为分数
        self.fitness = max(np.mean(self.fitness_values), 0)
    
        spinning_penalty = self.evaluate_spinning_penalty()

        if spinning_penalty == 1:
            return 0  # 如果自旋时间过长，适应度直接为 0

        return self.fitness
        

    def evaluate_line_tracking(self):
        # 直接使用地面传感器的读数
        left = self.left_ir.getValue()
        center = self.center_ir.getValue()
        right = self.right_ir.getValue()

        # 假设地面传感器的最大读数是 1000
        max_sensor_value = 1000

        # 计算每个传感器的得分（越接近中心线得分越高）
        left_score = (max_sensor_value - left) / max_sensor_value
        center_score = (max_sensor_value - center) / max_sensor_value
        right_score = (max_sensor_value - right) / max_sensor_value

        # 综合三个传感器的得分
        line_tracking_score = (left_score + center_score + right_score) / 3

        return line_tracking_score

    def evaluate_spinning_penalty(self):
        # 如果两个轮子的速度差异较大，则返回较大的惩罚值
        speed_difference = abs(self.velocity_left - self.velocity_right)
        penalty = speed_difference / (2 * self.max_speed) * 100
        return min(penalty, 1)  # 确保惩罚值不超过 1

    def evaluate_velocity_score(self):
        average_speed = (abs(self.velocity_left) + abs(self.velocity_right)) / 2
        return average_speed / self.max_speed
        
    def evaluate_out_of_line_penalty(self):
        left = self.left_ir.getValue()
        center = self.center_ir.getValue()
        right = self.right_ir.getValue()
        # 检查是否偏离路径
        if all(760 <= sensor_value <= 770 for sensor_value in [left, center, right]):
            return 0.5  # 偏离路径的惩罚分数
        return 0
       
    def evaluate_collision_penalty(self):
        # 设定碰撞阈值
        collision_threshold = 0.4
    
        # 检查所有传感器
        for i in range(8):  # 假设有8个距离传感器
            sensor_value = self.proximity_sensors[i].getValue()
            if sensor_value > collision_threshold:
                return 0.6  # 检测到碰撞，返回惩罚值
        return 1  # 没有检测到碰撞，不给予惩罚
        
    ######################################   
```

第四代

```
######################################
    ######################################
    #   在这里修改fitness函数
    ######################################
    ######################################
    
    def calculate_fitness(self):
        # 巡线得分
        line_tracking_score = self.evaluate_line_tracking()
    
        # 自旋惩罚
        spinning_penalty = self.evaluate_spinning_penalty()
    
        # 速度得分
        speed_score = self.evaluate_velocity_score()
    
        # 碰撞惩罚
        collision_penalty = self.evaluate_collision_penalty()
    
        # 线外扣分
        out_of_line_penalty = self.evaluate_out_of_line_penalty()
    
        # 计算最终适应度：乘积形式结合各个得分，并确保适应度不为负值
        combined_fitness = line_tracking_score * max(1 - spinning_penalty, 0) * speed_score * max(1 - collision_penalty, 0) * max(1 - out_of_line_penalty, 0)
        self.fitness_values.append(combined_fitness)
    
        # 计算平均适应度，并转换为分数
        self.fitness = max(np.mean(self.fitness_values), 0)
    
        return self.fitness
        

    def evaluate_line_tracking(self):
        # 直接使用地面传感器的读数
        left = self.left_ir.getValue()
        center = self.center_ir.getValue()
        right = self.right_ir.getValue()

        # 假设地面传感器的最大读数是 1000
        max_sensor_value = 1000

        # 计算每个传感器的得分（越接近中心线得分越高）
        left_score = (max_sensor_value - left) / max_sensor_value
        center_score = (max_sensor_value - center) / max_sensor_value
        right_score = (max_sensor_value - right) / max_sensor_value

        # 综合三个传感器的得分
        line_tracking_score = (left_score + center_score + right_score) / 3

        return line_tracking_score

    def evaluate_spinning_penalty(self):
        # 如果两个轮子的速度差异较大，则返回较大的惩罚值
        speed_difference = abs(self.velocity_left - self.velocity_right)
        penalty = speed_difference / (2 * self.max_speed) * 100
        return min(penalty, 1)  # 确保惩罚值不超过 1

    def evaluate_velocity_score(self):
        average_speed = (abs(self.velocity_left) + abs(self.velocity_right)) / 2
        return average_speed / self.max_speed
        
    def evaluate_out_of_line_penalty(self):
        left = self.left_ir.getValue()
        center = self.center_ir.getValue()
        right = self.right_ir.getValue()
        # 检查是否偏离路径
        if all(760 <= sensor_value <= 770 for sensor_value in [left, center, right]):
            return 1  # 偏离路径的惩罚分数
        return 0
       
    def evaluate_collision_penalty(self):
        # 设定碰撞阈值
        collision_threshold = 0.4
    
        # 检查所有传感器
        for i in range(8):  # 假设有8个距离传感器
            sensor_value = self.proximity_sensors[i].getValue()
            if sensor_value > collision_threshold:
                return 1  # 检测到碰撞，返回惩罚值
        return 0  # 没有检测到碰撞，不给予惩罚
        
    ###################################### 
```

第四代

```
def evaluate_line_tracking(self):
        # 直接使用地面传感器的读数
        left = self.left_ir.getValue()
        center = self.center_ir.getValue()
        right = self.right_ir.getValue()

        # 假设地面传感器的最大读数是 1000
        max_sensor_value = 1000

        # 计算每个传感器的得分（越接近中心线得分越高）
        left_score = (max_sensor_value - left) / max_sensor_value
        center_score = (max_sensor_value - center) / max_sensor_value
        right_score = (max_sensor_value - right) / max_sensor_value

        # 综合三个传感器的得分
        line_tracking_score = (left_score + center_score + right_score) / 3

        return line_tracking_score
```

第五代

```
    ######################################
    ######################################
    #   在这里修改fitness函数
    ######################################
    ######################################
    
    def calculate_fitness(self):
        # 巡线得分
        line_tracking_score = self.evaluate_line_tracking()
    
        # 自旋惩罚
        spinning_penalty = self.evaluate_spinning_penalty()
    
        # 速度得分
        speed_score = self.evaluate_velocity_score()
    
        # 碰撞惩罚
        collision_penalty = self.evaluate_collision_penalty()
    
        # 线外扣分
        out_of_line_penalty = self.evaluate_out_of_line_penalty()
    
        # print("line:",line_tracking_score,"spin:",spinning_penalty)
        # print("speed:",speed_score,"collision:",collision_penalty)
        # print("out_of_line_penalty",out_of_line_penalty)
    
        # 计算最终适应度：乘积形式结合各个得分，并确保适应度不为负值
        combined_fitness = speed_score * 0.5 + line_tracking_score * max(1 - spinning_penalty, 0) * max(1 - collision_penalty, 0) * max(1 - out_of_line_penalty, 0)
        self.fitness_values.append(combined_fitness)
    
        # 计算平均适应度，并转换为分数
        self.fitness = max(np.mean(self.fitness_values), 0)
    
        return self.fitness
        
    def evaluate_line_tracking(self):
        # 直接使用地面传感器的读数
        left = self.left_ir.getValue()
        center = self.center_ir.getValue()
        right = self.right_ir.getValue()
    
        # 设定扫描范围
        lower_bound = 760
        upper_bound = 770
    
        # 计算在范围内的传感器数量
        sensors_in_range = sum(lower_bound <= sensor_value <= upper_bound for sensor_value in [left, center, right])
    
        # 如果有两个或更多传感器在范围内，返回1，否则返回0
        if sensors_in_range >= 2:
            return 0
        else:
            return 1
    
    def evaluate_spinning_penalty(self):
        # 如果两个轮子的速度差异较大，则返回较大的惩罚值
        # speed_difference = abs(self.velocity_left - self.velocity_right)
        
        # penalty = speed_difference / (2 * self.max_speed) * 1000
        if ( abs(self.velocity_left) < 0.2*abs(self.velocity_right) ):
            return 1;
        if ( 0.2*abs(self.velocity_left) > abs(self.velocity_right) ):
            return 1;
        return 0  # 确保惩罚值不超过 1

    def evaluate_velocity_score(self):
        average_speed = (abs(self.velocity_left) + abs(self.velocity_right)) / 2
        return average_speed / self.max_speed
        
    def evaluate_out_of_line_penalty(self):
        left = self.left_ir.getValue()
        center = self.center_ir.getValue()
        right = self.right_ir.getValue()
        
        # print(left,center,right)
        
        # 检查每个传感器是否在指定范围内
        sensors_in_range = sum(290 <= sensor_value <= 350 for sensor_value in [left, center, right])
    
        # 如果少于两个传感器在指定范围内，则判断为偏离路径并返回惩罚分数
        if sensors_in_range < 2:
            return 5  # 偏离路径的惩罚分数
        return 0
       
    def evaluate_collision_penalty(self):
        # 设定碰撞阈值
        collision_threshold = 100
        save_value = 0
        # 检查所有传感器
        for i in range(8):  # 假设有8个距离传感器
            sensor_value = self.proximity_sensors[i].getValue()
            # print("sensor_value",sensor_value)
            if sensor_value > collision_threshold:
                save_value = 1  # 检测到碰撞，返回惩罚值
        return save_value  # 没有检测到碰撞，不给予惩罚
        
    ######################################    
```

备份：

```
    ######################################
    ######################################
    #   在这里修改fitness函数
    ######################################
    ######################################
    
    def calculate_fitness(self):
        # 巡线得分
        line_tracking_score = self.evaluate_line_tracking()
    
        # 自旋惩罚
        spinning_penalty = self.evaluate_spinning_penalty()
    
        # 速度得分
        speed_score = self.evaluate_velocity_score()
    
        # 碰撞惩罚
        collision_penalty = self.evaluate_collision_penalty()
    
        # 线外扣分
        out_of_line_penalty = self.evaluate_out_of_line_penalty()
    
        mark_score_1 = self.mark_score()
        # print("line:",line_tracking_score,"spin:",spinning_penalty)
        # print("speed:",speed_score,"collision:",collision_penalty)
        # print("out_of_line_penalty",out_of_line_penalty)
    
        # 计算最终适应度：乘积形式结合各个得分，并确保适应度不为负值
        combined_fitness = mark_score_1 + speed_score * 0.5 + line_tracking_score * max(1 - spinning_penalty, 0) * max(1 - collision_penalty, 0) * max(1 - out_of_line_penalty, 0)
        self.fitness_values.append(combined_fitness)
    
        # 计算平均适应度，并转换为分数
        self.fitness = max(np.mean(self.fitness_values), 0)
    
        return self.fitness
        
    def evaluate_line_tracking(self):
        # 直接使用地面传感器的读数
        left = self.left_ir.getValue()
        center = self.center_ir.getValue()
        right = self.right_ir.getValue()
    
        # 设定扫描范围
        lower_bound = 760
        upper_bound = 770
    
        # 计算在范围内的传感器数量
        sensors_in_range = sum(lower_bound <= sensor_value <= upper_bound for sensor_value in [left, center, right])
    
        # 如果有两个或更多传感器在范围内，返回1，否则返回0
        if sensors_in_range >= 2:
            return 0
        else:
            return 1
    
    def evaluate_spinning_penalty(self):
        # 如果两个轮子的速度差异较大，则返回较大的惩罚值
        # speed_difference = abs(self.velocity_left - self.velocity_right)
        
        # penalty = speed_difference / (2 * self.max_speed) * 1000
        if ( abs(self.velocity_left) < 0.2*abs(self.velocity_right) ):
            return 1;
        if ( 0.2*abs(self.velocity_left) > abs(self.velocity_right) ):
            return 1;
        return 0  # 确保惩罚值不超过 1

    def evaluate_velocity_score(self):
        average_speed = (abs(self.velocity_left) + abs(self.velocity_right)) / 2
        return average_speed / self.max_speed
        
    def mark_score(self):    
        left = self.left_ir.getValue()
        center = self.center_ir.getValue()
        right = self.right_ir.getValue()
        
        # print(left,center,right)
        
        # 检查每个传感器是否在指定范围内
        sensors_in_range = sum(480 <= sensor_value <= 500 for sensor_value in [left, center, right])
    
        # 如果少于两个传感器在指定范围内，则判断为偏离路径并返回惩罚分数
        if sensors_in_range >= 1:
            return 1  # 偏离路径的惩罚分数
        return 0
        
    def evaluate_out_of_line_penalty(self):
        left = self.left_ir.getValue()
        center = self.center_ir.getValue()
        right = self.right_ir.getValue()
        
        # print(left,center,right)
        
        # 检查每个传感器是否在指定范围内
        sensors_in_range = sum(290 <= sensor_value <= 350 for sensor_value in [left, center, right])
    
        # 如果少于两个传感器在指定范围内，则判断为偏离路径并返回惩罚分数
        if sensors_in_range < 2:
            return 5  # 偏离路径的惩罚分数
        return 0
       
    def evaluate_collision_penalty(self):
        # 设定碰撞阈值
        collision_threshold = 100
        save_value = 0
        # 检查所有传感器
        for i in range(8):  # 假设有8个距离传感器
            sensor_value = self.proximity_sensors[i].getValue()
            # print("sensor_value",sensor_value)
            if sensor_value > collision_threshold:
                save_value = 1  # 检测到碰撞，返回惩罚值
        return save_value  # 没有检测到碰撞，不给予惩罚
        
    ######################################   

```

第五代：

```
######################################
    ######################################
    #   在这里修改fitness函数
    ######################################
    ######################################
    
    def evaluate_line_tracking(self):
        left_value = self.left_ir.getValue()
        center_value = self.center_ir.getValue()
        right_value = self.right_ir.getValue()
        # #
        # #
        if self.counter % 32 == 0:
            print("Ground sensor readings:", left_value, center_value, right_value)
            self.counter = 0  # 重置计数器
        # #
        # #
        self.counter += 1
    
        # 假设地面传感器的读数已经储存在self.inputs中
        ground_sensor_readings = self.inputs[-3:]  # 假设最后3个输入是地面传感器
        line_tracking_score = sum(ground_sensor_readings) / len(ground_sensor_readings)  # 平均值
        normalized_score = line_tracking_score / 4095  # 假设最大值为4095
        return normalized_score

    def evaluate_stable_movement(self):
        # 计算左右轮速度的平均值
        average_speed = (abs(self.velocity_left) + abs(self.velocity_right)) / 2
        # 避免自旋转：左右轮速度差异的惩罚
        speed_difference = abs(self.velocity_left - self.velocity_right)
        spinning_penalty = speed_difference / (2 * self.max_speed)  # 归一化到0到1的范围
        # 综合得分
        stable_movement_score = average_speed / self.max_speed - spinning_penalty
        return max(0, stable_movement_score)  # 确保分数不为负

    def calculate_fitness(self):
        line_tracking_score = self.evaluate_line_tracking()
        stable_movement_score = self.evaluate_stable_movement()

        # 综合得分
        combined_fitness = line_tracking_score + stable_movement_score
        
        if combined_fitness>0.6:
            combined_fitness = 0
        # print(combined_fitness)
        
        self.fitness_values.append(combined_fitness)
        self.fitness = np.mean(self.fitness_values)
        
    ######################################   
```

第六代：

```
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
        
        # print("left_value",left_value,center_value,right_value)
        
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
        
        normalized_score = -1 * score_point / 3 # 分数范围0-1
        
        return normalized_score


    def evaluate_stable_movement(self):
        # 计算左右轮速度的平均值
        max_speed = max(self.velocity_left,self.velocity_right) + 0.000001
        
        average_speed = (abs(self.velocity_left) + abs(self.velocity_right)) / 2
        
        # 避免自旋转：左右轮速度差异的惩罚
        speed_difference = (abs(self.velocity_left - self.velocity_right)) / max_speed
        
        # print("average_speed",average_speed,speed_difference)
        if ( 0<= speed_difference <=0.6 ):
            normalized_score = average_speed
        else:
            normalized_score = 0
        
        return normalized_score  # 确保分数不为负

    def calculate_fitness(self):
        line_tracking_score = self.evaluate_line_tracking()
        outline_tracking_score = self.evaluate_outline_tracking()
        stable_movement_score = self.evaluate_stable_movement()

        # print(line_tracking_score,outline_tracking_score)
        # 综合得分
        combined_fitness = (line_tracking_score + outline_tracking_score) * stable_movement_score
        
        # if combined_fitness>0.6:
            # combined_fitness = 0
        # print(combined_fitness)
        
        self.fitness_values.append(combined_fitness)
        self.fitness = np.mean(self.fitness_values)
        
        
    ######################################    
```

第七代：

```
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
        
        # print("left_value",left_value,center_value,right_value)
        
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
        
        normalized_score = -1 * score_point / 3 # 分数范围0-1
        
        return normalized_score


    def evaluate_stable_movement(self):
        # 计算左右轮速度的平均值
        max_speed = max(self.velocity_left,self.velocity_right) + 0.000001
        
        average_speed = (abs(self.velocity_left) + abs(self.velocity_right)) / 2
        
        # 避免自旋转：左右轮速度差异的惩罚
        speed_difference = (abs(self.velocity_left - self.velocity_right)) / max_speed
        
        # print("average_speed",average_speed,speed_difference)
        if ( 0<= speed_difference <=0.6 ):
            normalized_score = average_speed
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
        
        if (lower <= left_value <= higher):
             score_point = score_point + 1
        if (lower <= center_value <= higher):
             score_point = score_point + 1
        if (lower <= right_value <= higher):
             score_point = score_point + 1
        # print("score_point",score_point)
        if (score_point>=2):
            score_point = 90
        else:
            score_point = 0
        
        normalized_score = score_point / 3 # 分数范围0-1
        
        return normalized_score
    
    def calculate_fitness(self):
        line_tracking_score = self.evaluate_line_tracking()
        outline_tracking_score = self.evaluate_outline_tracking()
        stable_movement_score = self.evaluate_stable_movement()
        bonus_value = self.bonus_value()
        # print(line_tracking_score,outline_tracking_score)
        # 综合得分
        combined_fitness = bonus_value + (line_tracking_score + outline_tracking_score) * stable_movement_score
        
        # if combined_fitness>0.6:
            # combined_fitness = 0
        # print(combined_fitness)
        
        self.fitness_values.append(combined_fitness)
        self.fitness = np.mean(self.fitness_values)
        
        
    ######################################    
```

第八代：

```
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
        
        # print("left_value",left_value,center_value,right_value)
        
        lower = 0.29
        higher = 0.35
        score_point = 0
        
        if (lower <= left_value <= higher):
             score_point = score_point + 1
        if (lower <= center_value <= higher):
             score_point = score_point + 1
        if (lower <= right_value <= higher):
             score_point = score_point + 1
        
        normalized_score = 3 * score_point / 3 # 分数范围0-1
        
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
        
        normalized_score = -1 * score_point / 3 # 分数范围0-1
        
        return 0


    def evaluate_stable_movement(self):
        # 计算左右轮速度的平均值
        max_speed = max(self.velocity_left,self.velocity_right) + 0.000001
        
        average_speed = (abs(self.velocity_left) + abs(self.velocity_right)) / 2
        
        # 避免自旋转：左右轮速度差异的惩罚
        speed_difference = (abs(self.velocity_left - self.velocity_right)) / max_speed
        
        # print("average_speed",average_speed,speed_difference)
        if ( 0<= speed_difference <=0.6 ):
            normalized_score = average_speed
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
        
        if (lower <= left_value <= higher):
             score_point = score_point + 1
        if (lower <= center_value <= higher):
             score_point = score_point + 1
        if (lower <= right_value <= higher):
             score_point = score_point + 1
        # print("score_point",score_point)
        if (score_point>=2):
            score_point = 90
        else:
            score_point = 0
        
        normalized_score = score_point / 3 # 分数范围0-1
        
        return normalized_score
    
    def keep_away_bonus(self):
        score_point = 0
        for i in range(8):
            temp = self.proximity_sensors[i].getValue() / 2000
            if ( 0.6<=temp<=0.8 ):
                score_point = score_point + 50
            elif ( temp>0.8 ):
                score_point = score_point - 50
                
        normalized_score = score_point / 3 
        
        return normalized_score
    
    def calculate_fitness(self):
        line_tracking_score = self.evaluate_line_tracking()
        outline_tracking_score = self.evaluate_outline_tracking()
        stable_movement_score = self.evaluate_stable_movement()
        bonus_value = self.bonus_value()
        keep_away_bonus = self.keep_away_bonus()
        # print(line_tracking_score,outline_tracking_score)
        # 综合得分
        combined_fitness = keep_away_bonus * 0.01 + bonus_value + (line_tracking_score + outline_tracking_score) * stable_movement_score
        
        # if combined_fitness>0.6:
            # combined_fitness = 0
        # print(combined_fitness)
        
        self.fitness_values.append(combined_fitness)
        self.fitness = np.mean(self.fitness_values)
        
        
    ######################################   
```

第九代：

```
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
        
        # print("left_value",left_value,center_value,right_value)
        
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
```

