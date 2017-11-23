import time
import os
import sys
import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
import math
import pickle

class Confidence:
  confidence_length = 24
  def __init__(self):
    self.value = 1
  def add_confidence(self, conf, distance):
    val = self.value 
    if conf is not None:
      val = max(val, conf.value)
    self.value = val + (self.confidence_length - distance) / float(self.confidence_length)

class CellValue:
  def __init__(self, cr = None):
    self.options = {}
    self.cell = cr

  def __str__(self):
    if self.cell is not None:
      return str(self.cell)
    if len(self.options) == 0:
      return "-"
    return ",".join([str(self.options[a]) for a in self.options])

  def choose_cell(self):
    self.cell = self.confident_cell()

  def confident_cell(self):
    if len(self.options) == 0:
      return None      
    return max(self.options.items(), key=lambda x: x[1].conf_value())[1]

  def add_option(self, cell_rec):
    if cell_rec.cell in self.options:
      self.options[cell_rec.cell].increase_confidence()
    else:
      self.options[cell_rec.cell] = cell_rec

  def connect(self, cv, pos):
    for crk in self.options:
      cr = self.options[crk]
      for cr2k in cv.options:
        cr2 = cv.options[cr2k]
        if cr.cell == cr2.cell:
          if cr.confidence is None:
            cr.confidence = Confidence()
          cr.confidence.add_confidence(cr2.confidence, pos)
          cr2.confidence = cr.confidence

  def calc_distance(self, detector, cv, pos):
    for crk in self.options:
      cr = self.options[crk]
      for cr2k in cv.options:
        cr2 = cv.options[cr2k]
        cr2.calc_distance(detector, cr, pos)  

class CellRec:
  def __init__(self, cell, cam = -1):
    self.cell = cell
    self.cam = cam
    self.confidence = Confidence()

  def increase_confidence(self):
    self.confidence.value = self.confidence.value * 2

  def rect_center(rect):
    cx = int(rect[0] + (rect[2] - rect[0])/2)
    cy = int(rect[1] + (rect[3] - rect[1])/2)
    return (cx,cy)
    
  def abs_dist(dx,dy):
    return math.sqrt(dx*dx+dy*dy) 

  def calc_distance(self, detector, cr, pos):
    if self.cell == cr.cell:
      self.dx = 0
      self.dy = 0
      self.prev_pos = pos
      self.closest_cell = cr.cell
    else:
      rect1 = detector.cams_map[self.cell][self.cam]
      if self.cam in detector.cams_map[cr.cell]:
        rect2 = detector.cams_map[cr.cell][self.cam]
        center1 = CellRec.rect_center(rect1)
        center2 = CellRec.rect_center(rect2)
        dx = center1[0] - center2[0]
        dy = center1[1] - center2[1]
        if ((self.dx == -1) or (CellRec.abs_dist(dx,dy) < CellRec.abs_dist(self.dx,self.dy))):
          self.dx = dx
          self.dy = dy
          self.prev_pos = pos
          self.closest_cell = cr.cell

  def conf_value(self):
    if self.confidence is None:
      return 0

    return self.confidence.value

  def __str__(self):
    return "{}".format(self.cell)

class Detector:
  delimiter = 4
  detection_columns = ['camera_view','frame','bounding_box_x','bounding_box_y','width','height',
       'ref','ref_score', 'team_a','team_a_score','team_b','team_b_score','keeper_a','keeper_a_score',
       'keeper_b','keeper_b_score','jersey_1','jersey_1_score','jersey_2','jersey_2_score','jersey_3',
       'jersey_3_score','jersey_4','jersey_4_score','jersey_5','jersey_5_score']
  def load_configs(self, room_path, detections_path, positions_path):
    self.read_room(room_path)
    self.read_detections(detections_path)
    self.read_positions(positions_path)

  def cell_to_rect(self,cell,camera_view): 
    if cell in self.cams_map:
      return self.cams_map[cell][camera_view]

  def row_to_cells_df(self, row):
    start_x = int(row['bounding_box_x'] / self.delimiter)
    start_y = int(row['bounding_box_y'] / self.delimiter)
    finish_x = int((row['width']+row['bounding_box_x']) / self.delimiter)
    finish_y = int((row['bounding_box_y']+row['height']) / self.delimiter)

    cells = []
    for i in range(start_x,finish_x+1):
      for j in range(start_y,finish_y+1):
        k = (i,j,row['camera_view'])
        if k in self.reverse_map:
          cells.append(self.reverse_map[k])

    return pd.DataFrame(cells)

  def row_to_cell(self, row, nlargest = 5):
    cam = row['camera_view']
    cells_df = self.row_to_cells_df(row)
    cells_score = cells_df.apply(pd.value_counts).fillna(0).sum(axis=1)

    if len(cells_score) == 0:
      return

    cv = CellValue()
    for cell in cells_score.nlargest(nlargest).index:    
      cv.add_option(CellRec(cell, cam))

    return cv    


  def rows_to_cell(self, rows, nlargest = 5):    
    res = pd.Series()
    cam = -1
    for index in rows.index:
      row = rows.loc[index]
      cam = row['camera_view']
      cells_df = self.row_to_cells_df(row)
      cells_score = cells_df.apply(pd.value_counts).fillna(0).sum(axis=1)
      res.add(cells_score, fill_value=0)

    if cam == -1 or len(res) == 0:
      return

    cv = CellValue()
    for cell in res.nlargest(nlargest).index:    
      cv.add_option(CellRec(cell, cam))

    return cv

  def row_to_cells_set(self, row):
    cells_df = self.row_to_cells_df(row)
    return set(np.unique(cells_df.values))

  def read_room(self,fname):
    with open(fname) as f:
      content = f.readlines()

    content = [x.strip() for x in content]
    header = content[0]
    content = content[1:]
    self.width, self.height, self.cams_count, self.cells_count = [int(a) for a in header.split(' ')[1:]]
    self.cams_map = dict()
    self.reverse_map = {}
    for line in content:
      arr = line.split(' ')
      if len(arr) == 7:
        camera_view, cell, x1, y1, x2, y2 = [int(a) for a in arr[1:]]
        camera_view = camera_view + 1
        dx1,dx2,dy1,dy2 = int(x1/self.delimiter), int(x2/self.delimiter), int(y1/self.delimiter), int(y2/self.delimiter)
        for i in range(dx1,dx2+1):
          for j in range(dy1,dy2+1):
            self.reverse_map[(i,j,camera_view)] = cell
        if cell in self.cams_map:
          self.cams_map[cell][camera_view] = [x1, y1, x2, y2]
        else:
          self.cams_map[cell] = {camera_view: [x1, y1, x2, y2]}  

  def read_positions(self,fname):
    self.positions = []
    with open(fname) as f:
      content = f.readlines()
    content = [x.strip() for x in content][1:]
      
    for line in content:
      arr = line.split(' ')      
      index = int(arr[0])
      start_frame = int(arr[1])
      score = float(arr[3])
      
      cells = [int(a) for a in arr[4:]]      
      self.positions.append({'start_frame': start_frame, 'score': score, 'cells': cells})
    return

  def square(rect):
    return (rect[2]-rect[0]) * (rect[3]-rect[1])

  def find_record(self,cam, frame, rect, th=0.3):  
    ds = self.detections
    ds = ds[((ds.frame==frame) & (ds.camera_view==cam))]
    sq = Detector.square(rect)

    res = []
    for index, row in ds.iterrows():
      dx = (min(rect[2], row.bounding_box_x+row.width) - max(rect[0],row.bounding_box_x))
      dy = (min(rect[3], row.bounding_box_y+row.height) - max(rect[1],row.bounding_box_y))
      intersect_kof = (dx*dy) / sq
      if dx > 0 and dy > 0 and intersect_kof > th:
        res.append(row)

    return res  

  def get_number_from_row(self, row):    
    if row is None:
      return

    if row['team_b_score'] > row['team_a_score']:
      team = "B"
    else:
      team = "A"

    for index in range(1,6):
      jersey_key = 'jersey_{}'.format(index)
      if row[jersey_key + '_score'] > 0.5 and row[jersey_key] not in ['fv','ref']:
        return team + row[jersey_key]

    return None
        
  def find_number(self, frame, cell):    
    numbers = []
    for cam in self.cams_map[cell]:
      rect = self.cams_map[cell][cam]
      rows = self.find_record(cam, frame, rect)
      for row in rows:
        num = self.get_number_from_row(row)
        if num is not None:
          numbers.append((num,cam))
    return numbers

  def read_detections(self,fname):
    with open(fname, "r") as ins:
      array = []
      for line in ins:
        array.append(line.strip().split(','))
    self.detections = pd.DataFrame(array,columns=self.detection_columns)
    self.detections.frame = self.detections.frame.astype(int)
    self.detections.camera_view = self.detections.camera_view.astype(int)
    self.detections.bounding_box_x = self.detections.bounding_box_x.astype(int)
    self.detections.bounding_box_y = self.detections.bounding_box_y.astype(int)
    self.detections.width = self.detections.width.astype(int)
    self.detections.height = self.detections.height.astype(int)
    for index in range(1,6):
      jersey_score_key = 'jersey_{}_score'.format(index)
      self.detections[jersey_score_key] = self.detections[jersey_score_key].astype(float)
    self.detections['team_a_score'] = self.detections['team_a_score'].astype(float)
    self.detections['team_b_score'] = self.detections['team_b_score'].astype(float)
    
    self.detections.drop(['team_a','team_b', 'keeper_a','keeper_b','keeper_a_score','keeper_b_score','ref','ref_score'], axis=1,inplace=True)
    self.frames_count = self.detections.frame.max() + 1

  def drop_spare_tracks(self,df, detection_error_threshold = 0.1):    
    res = (df.shape[0] - df.isnull().sum())/df.shape[0]
    valid_players = res > detection_error_threshold
    df = df[res[valid_players].index]
    return df

  def add_top_cell_metric(self,df):
    for number in df.columns:
      for index in range(df.shape[0]):
        if pd.isnull(df.loc[index, number]):
          continue
        df.loc[index, number].choose_cell()
    return df

  def add_confidence_metric(self,df):
    for number in df.columns:
      for index in range(df.shape[0] - 1):
        if pd.isnull(df.loc[index, number]):
          continue
        cv = df.loc[index, number]
        for j in range(index+1, min(index+Confidence.confidence_length, df.shape[0])):
          if pd.isnull(df.loc[j, number]):
            continue
          cv_next = df.loc[j, number]
          cv.connect(cv_next, j - index)
    return df

  def fill_empty_cells_with_detections(self,df):
    for number in df.columns:
      player_df = df.loc[pd.isnull(df[number]),number]
      for index in player_df.index:
        df.loc[index, number] = self.get_cell_from_detections(index, number)
    return df

  def get_cell_from_detections(self, frame, number):
    team_key = 'team_{}_score'.format(number[0].lower())
    ds = self.detections
    rows = ds[((ds.frame==frame) & (ds[team_key]>0.5) & (ds.jersey_1==number[1:]))]
    if rows.shape[0] == 0:
      return
    cv = self.rows_to_cell(rows)
    return cv

  def fill_tracks_linear(self,df):
    for number in df.columns:

      start_index = -1
      index = 0

      while index < (df.index.shape[0]):
        if pd.notnull(df.loc[index, number]):          
          if start_index >= 0 and (index - start_index > 1):
            suggested_cells = self.suggest_route(df.loc[start_index,number],df.loc[index,number], index - start_index)
            if suggested_cells:
              df.loc[(start_index+1):(index-1),number] = suggested_cells
          start_index = index    
        index = index + 1    

    return df

  def suggest_route(self, start_cv, finish_cv, steps):
    if steps <= 1:
      return
    start_cr = start_cv.cell
    finish_cr = finish_cv.cell
    if (finish_cr is None) or (start_cr is None):
      return
    if start_cr.cam not in self.cams_map[finish_cr.cell]:
      return

    start_rect = self.cams_map[start_cr.cell][start_cr.cam]
    finish_rect = self.cams_map[finish_cr.cell][start_cr.cam]

    start_x = int(start_rect[0] / self.delimiter)
    start_y = int(start_rect[1] / self.delimiter)
    finish_x = int(finish_rect[0] / self.delimiter)
    finish_y = int(finish_rect[1] / self.delimiter)

    dx = finish_x - start_x
    dy = finish_y - start_y    

    step_x = float(dx) / steps
    step_y = float(dy) / steps

    cells = [self.reverse_map[(start_x + int(step_x*i),start_y+int(step_y*i),start_cr.cam)] for i in range(1,steps)]
    return [CellValue(CellRec(cell,start_cr.cam)) for cell in cells]

  def build_tracks_from_detections(self):
    results = pd.DataFrame()
    df = pd.DataFrame(index=range(self.frames_count))
    for index, row in self.detections.iterrows():
      number = self.get_number_from_row(row)
      if number:
        frame = row['frame']
        df.loc[frame, number] = self.row_to_cell(row)
    
    df = self.drop_spare_tracks(df)
    df = self.add_confidence_metric(df)
    df = self.add_top_cell_metric(df)    
    df = self.fill_tracks_linear(df)
    return df

  def build_tracks_from_positions(self):
    df = pd.DataFrame(index=range(self.frames_count))
    for positions in self.positions:
      start_frame = positions['start_frame']
      for (index,cell) in enumerate(positions['cells']):
        frame = start_frame+index
        numbers = self.find_number(frame, cell)
        for number, cam in numbers:        
          if (number not in df.columns) or pd.isnull(df.loc[frame, number]):
            cv = CellValue()
            df.loc[frame, number] = cv
          else:
            cv = df.loc[frame, number]          
          cv.add_option(CellRec(cell, cam))
    
    df = self.drop_spare_tracks(df)
    df = self.fill_empty_cells_with_detections(df)
    df = self.add_confidence_metric(df)
    df = self.add_top_cell_metric(df)
    df = self.fill_tracks_linear(df)
    return df

if __name__ == "__main__":
  detector = Detector()
  detector.load_configs("task_track_identification_resources/room.pom",
    "task_track_identification_resources/detections.dat",
    "task_track_identification_resources/mtp-out-vBGtestv1-50traj.trj",
    )  
  df = detector.build_tracks_from_detections()
  df.to_csv('tracks_from_detections.csv', index=True)