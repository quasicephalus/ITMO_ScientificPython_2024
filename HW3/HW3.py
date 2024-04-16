import openmeteo_requests
from datetime import datetime


class IncreaseSpeed():
  '''
  Iterator for increasing the speed with the default step of 10 km/h
  You can implement this one after Iterators FP topic

  Constructor params:  
    current_speed: a value to start with, km/h
    max_speed: a maximum possible value, km/h

  Make sure your iterator is not exceeding the maximum allowed value
  '''

  def __init__(self, current_speed: int, max_speed: int):
    # self.start_speed = current_speed
    self.current_speed = current_speed
    self.max_speed = max_speed
    self.step = 10

  def __iter__(self):
    # self.current_speed = self.start_speed
    return self
  
  def __next__(self):
    if self.current_speed + self.step <= self.max_speed:
        self.current_speed += self.step
        return self.current_speed
    else:
      raise StopIteration


class DecreaseSpeed():
  '''
  Iterator for decreasing the speed with the default step of 10 km/h
  You can implement this one after Iterators FP topic 

  Constructor params: 
    current_speed: a value to start with, km/h

  Make sure your iterator is not going below zero
  '''

  def __init__(self, current_speed: int):
    self.current_speed = current_speed
    self.step = 10

  def __iter__(self):
    return self
  
  def __next__(self):
    if self.current_speed - self.step >= 0:
        self.current_speed -= self.step
        return self.current_speed
    else:
      raise StopIteration

class Car():
  '''
  Car class. 
  Has a class variable for counting total amount of cars on the road (increased by 1 upon instance initialization).

  Constructor params:
    max_speed: a maximum possible speed, km/h
    current_speed: current speed, km/h (0 by default)
    state: reflects if the Car is in the parking or on the road

  Methods:
    accelerate: increases the speed using IncreaseSpeed() iterator either once or gradually to the upper_border
    brake: decreases the speed using DecreaseSpeed() iterator either once or gradually to the lower_border
    parking: if the Car is not already in the parking, removes the Car from the road
    total_cars: show the total amount of cars on the road
    show_weather: shows the current weather conditions
  '''
  n_cars = 0

  def __init__(self, max_speed: int, current_speed=0):
    Car.n_cars += 1
    self.max_speed = max_speed
    self.current_speed = current_speed
    self.state = False # False if NOT parked (on the road) True if parked (not on the road)
  

  def accelerate(self, upper_border=None):
    # check for state 
    # create an instance of IncreaseSpeed iterator
    # check if smth passed to upper_border and if it is valid speed value
    # if True, increase the speed gradually iterating over your increaser until upper_border is met
    # print a message at each speed increase
    # else increase the speed once 
    # return the message with current speed
    if self.state:
      return 'Can\'t accelerate when car is parked.'
    
    speediter = IncreaseSpeed(self.current_speed, self.max_speed)

    if upper_border == None:
      upper_border = self.max_speed
    if upper_border < self.current_speed or upper_border > self.max_speed:
      print('Upper border is incorrect. It should be either None or integer >= current speed and <= max speed.')
      self.current_speed = next(speediter)

    else:
      for speed in speediter:
        if speed > upper_border:
          break
        self.current_speed = speed
        print(f'Speed is now {speed} km/h.')
    return f'Acceleration is over, current speed is {self.current_speed} km/h'
    


  def brake(self, lower_border=None):
    # create an instance of DecreaseSpeed iterator
    # check if smth passed to lower_border and if it is valid speed value
    # if True, decrease the speed gradually iterating over your decreaser until lower_border is met
    # print a message at each speed decrease
    # else increase the speed once 
    # return the message with current speed
    if self.state:
      return 'Car is parked.'
    
    speediter = DecreaseSpeed(self.current_speed)

    if lower_border == None:
      lower_border = 0
    if lower_border > self.current_speed or lower_border < 0:
      print('Lower border is incorrect. It should be either None or integer <= current speed and >= 0.')
      self.current_speed = next(speediter)

    else:
      for speed in speediter:
        if speed < lower_border:
          break
        self.current_speed = speed
        print(f'Speed is now {speed} km/h.')
    return f'Braking is over, current speed is {self.current_speed} km/h'

  # the next three functions you have to define yourself
  # one of the is class method, one - static and one - regular method (not necessarily in this order, it's for you to think)

  def parking(self): # This will be the instance method
    # gets car off the road (use state and class variable)
    # check: should not be able to move the car off the road if it's not there
    if not self.state:
      self.state = True
      self.current_speed = 0
      Car.n_cars -= 1
      return 'Car is parked.'
    return 'Car is already parked.'
  
  @classmethod
  def total_cars(cls): # This will be a class method
    print(f'{cls.n_cars} cars on the road.')

  @staticmethod
  def show_weather(): # This will be a staticmethod
    # displays weather conditions
    openmeteo = openmeteo_requests.Client()
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
    "latitude": 59.9386, # for St.Petersburg
    "longitude": 30.3141, # for St.Petersburg
    "current": ["temperature_2m", "apparent_temperature", "rain", "wind_speed_10m"],
    "wind_speed_unit": "ms",
    "timezone": "Europe/Moscow"
    }

    response = openmeteo.weather_api(url, params=params)[0]

    current = response.Current()
    current_temperature_2m = current.Variables(0).Value()
    current_apparent_temperature = current.Variables(1).Value()
    current_rain = current.Variables(2).Value()
    current_wind_speed_10m = current.Variables(3).Value()

    print(f"Current time: {datetime.fromtimestamp(current.Time()+response.UtcOffsetSeconds())} {response.TimezoneAbbreviation().decode()}")
    print(f"Current temperature: {round(current_temperature_2m, 0)} C")
    print(f"Current apparent_temperature: {round(current_apparent_temperature, 0)} C")
    print(f"Current rain: {current_rain} mm")
    print(f"Current wind_speed: {round(current_wind_speed_10m, 1)} m/s")


if __name__ == '__main__':
  car = Car(current_speed=0, max_speed=200)
  print(car.accelerate(200))
  print(car.brake(13))
  print(car.accelerate(200))
  car.parking()
  print(car.accelerate(199))
  Car.total_cars()
  Car.show_weather()

