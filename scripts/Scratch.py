df_piv = data_frame.pivot('X','Y','Bz')
xi,yi = np.meshgrid(df_piv.index,df_piv.columns)
zi = df_piv.values
old_points = zip(xi.ravel(),yi.ravel())
grid_x,grid_y = np.meshgrid(data_frame_interp.X, data_frame_interp.Y)
grid_z1 = griddata(old_points, zi.ravel(1), (grid_x, grid_y), method='linear')

data_interp = np.array([grid_x, grid_y, grid_z1]).reshape(3, -1).T





%run MagFieldPlotter.py
data_frame_grid
%run MagFieldPlotter.py
data_frame_grid
data_frame_grid.head()
%run MagFieldPlotter.py
data_frame_grid.head()
%run MagFieldPlotter.py
data_frame_grid.head()
df_piv = data_frame.pivot('X','Y','Bz')
df_pv
df_piv
x = df_piv.index.values
y = df_piv.column.values
y = df_piv.columns.values
z = df_piv.values
df_piv_interp = data_frame_grid.pivot('X','Y','Bz')
xx,yy = np.meshgrid(x,y)
f = interpolate.interp2d(x, y, z, kind='cubic')
from scipy import interpolate
f = interpolate.interp2d(x, y, z, kind='cubic')
f = interpolate.interp2d(x, y, z.T, kind='cubic')
xnew = df_piv_interp.index.values
ynew = df_piv_interp.columns.values
znew = f(xnew,ynew)
znew
data_interp = np.array([xnew, ynew, znew]).reshape(3, -1).T
xxnew,yynew = np.meshgrid(xnew,ynew)
data_interp = np.array([xxnew, yynew, znew]).reshape(3, -1).T
data_interp
len(data_interp)
len(data_interp.T)
data_interp
data_frame
data_frame_grid
data_frame_new = pd.DataFrame(data_interp, columns=['X','Y','Bz'])
data_frame_new
data_frame_newdef make_theta(self,row):
  def make_theta(row):
        return np.arctan2(row['Y'],row['X'])
      def make_r(row):
            return np.sqrt(row['X']**2+row['Y']**2)
          data_frame_new['Theta'] = data_frame_new.apply(make_theta,axis=1)
          data_frame_new['R'] = data_frame_new.apply(make_r,axis=1)
          data_frame_new[data_frame_new.R == 201.556444]
          df_tmp = data_frame_new[data_frame_new.R == 201.556444]
          plt.scatter(df_tmp.Theta, df_tmp.Bz)

