import torch 
from torch import nn
import pandas as pd
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.preprocessing import StandardScaler

data=pd.read_csv('diabetes.csv')

'''make report for visualize and stat'''
# profile=ProfileReport(data,)
# profile.to_file('report.html')
X=data.drop('Outcome',axis=1)
y=data['Outcome']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=159)



scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


y_train=y_train.to_numpy()
y_test=y_test.to_numpy()

'''transform data to compatible with pytorch'''
X_train = torch.from_numpy(X_train).type(torch.float)
y_train = torch.from_numpy(y_train).type(torch.float)
X_test = torch.from_numpy(X_test).type(torch.float)
y_test = torch.from_numpy(y_test).type(torch.float)


class DNN_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(8,10)
        self.layer_2 = nn.Linear(10,20)
        self.layer_3 = nn.Linear(20,1)
        self.relu = nn.ReLU() 

    def forward(self, x):
        x=self.relu(self.layer_1(x))
        x=self.relu(self.layer_2(x))
        x=self.layer_3(x)
        return x
    #cant return the sigmoid(x) because it will affect the loss function
model_0=DNN_model()
loss_fn=nn.BCEWithLogitsLoss() 
optimizer=torch.optim.SGD(params=model_0.parameters(), lr=0.1)


torch.manual_seed(159)
epochs=1000
for epoch in range(epochs):

    y_logits=model_0(X_train).squeeze()
    y_predict=torch.round(torch.sigmoid(y_logits)) #threshold is 0,5 can be change to increase preicsion or recall...
    
    
    loss = loss_fn(y_logits, y_train) 
    
    
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    # testing phase
    model_0.eval()
    with torch.inference_mode():
      
      test_logits=model_0(X_test).squeeze()
      test_predict=torch.round(torch.sigmoid(test_logits)) #similarly, could control threshold
     

      test_loss=loss_fn(test_logits, y_test)
      test_acc=accuracy_score(y_test,test_predict)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.4}")