load Speed70 % curr data for rotational velocity rad/s : 70
InputData70=[curr.th_d ; curr.th_dd];
OutputData70=curr.tau_ff;
load Speed71 % curr data for rotational velocity rad/s : 71
InputData71=[curr.th_d ; curr.th_dd];
OutputData71=curr.tau_ff;
load Speed72 % curr data for rotational velocity rad/s : 72
InputData72=[curr.th_d ; curr.th_dd];
OutputData72=curr.tau_ff;
load Speed73 % curr data for rotational velocity rad/s : 73
InputData73=[curr.th_d ; curr.th_dd];
OutputData73=curr.tau_ff;
load Speed74 % curr data for rotational velocity rad/s : 74
InputData74=[curr.th_d ; curr.th_dd];
OutputData74=curr.tau_ff;
load Speed75 % curr data for rotational velocity rad/s : 75
InputData75=[curr.th_d ; curr.th_dd];
OutputData75=curr.tau_ff;
load Speed76 % curr data for rotational velocity rad/s : 76
InputData76=[curr.th_d ; curr.th_dd];
OutputData76=curr.tau_ff;
load Speed77 % curr data for rotational velocity rad/s : 77
InputData77=[curr.th_d ; curr.th_dd];
OutputData77=curr.tau_ff;
load Speed78 % curr data for rotational velocity rad/s : 78
InputData78=[curr.th_d ; curr.th_dd];
OutputData78=curr.tau_ff;
load Speed79 % curr data for rotational velocity rad/s : 79
InputData79=[curr.th_d ; curr.th_dd];
OutputData79=curr.tau_ff;
load Speed80 % curr data for rotational velocity rad/s : 80
InputData80=[curr.th_d ; curr.th_dd];
OutputData80=curr.tau_ff;

x=[InputData70 InputData71 InputData72 InputData73 InputData74 InputData75 InputData76 InputData77 InputData78 InputData79 InputData80]; % inputs for neural network
tar=[OutputData70 OutputData71 OutputData72 OutputData73 OutputData74 OutputData75 OutputData76 OutputData77 OutputData78 OutputData79 OutputData80]; % targets for neural network
net = feedforwardnet(10); % Feedforward neural network with one hidden layer of size 10
[net,tr] = train(net,x,tar); % Train neural network
tau_ff = net(x); % Outputs / Estimates targets (feedforward torque) of neural network
NeuralNetworkQ3 = net;
save NeuralNetworkQ3