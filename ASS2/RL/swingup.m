function [par, ta, xa] = swingup(par)

    par.simtime = 10;     % Trial length
    par.simstep = 0.05;   % Simulation time step
    par.maxtorque = 1.5;  % Maximum applicable torque
    
    
    if strcmp(par.run_type, 'learn')
        %%
        % Obtain SARSA parameters
        par = get_parameters(par);
        
		% TOREVIEW: Initialize the outer loop
        Q = init_Q(par);

        % Initialize bookkeeping (for plotting only)
        ra = zeros(par.trials, 1);
        tta = zeros(par.trials, 1);
        te = 0;

        % Outer loop: trials
        for ii = 1:par.trials

            % TOREVIEW: Initialize the inner loop
            x = swingup_initial_state();
            s = discretize_state(x, par);
            a = execute_policy(Q, s, par);

            % Inner loop: simulation steps
            for tt = 1:ceil(par.simtime/par.simstep)
                
                % TOREVIEW: obtain torque
                u = take_action(a, par);
                
                % Apply torque and obtain new state
                % x  : state (input at time t and output at time t+par.simstep)
                % u  : torque
                % te : new time
                [te, x] = body_straight([te te+par.simstep],x,u,par);

                % TOREVIEW: learn
                % use s for discretized state
                sP = discretize_state(x, par);
                r = observe_reward(a, sP, par);
                
                aP = execute_policy(Q, sP, par);
                
                
                Q = update_Q(Q, s, a, r, sP, aP, par);
                
                s = sP;
                a = aP;
                
                % Keep track of cumulative reward
                ra(ii) = ra(ii)+r;

                % TOREVIEW: check termination condition
                t = is_terminal(s, par);
                if t 
                    break
                end
                
            end

            tta(ii) = tta(ii) + tt*par.simstep;

            % Update plot every ten trials
            if rem(ii, 10) == 0
                plot_Q(Q, par, ra, tta, ii);
                drawnow;
            end
        end
        
        % save learned Q value function
        par.Q = Q;
 
    elseif strcmp(par.run_type, 'test')
        %%
        % Obtain SARSA parameters
        par = get_parameters(par);
        
        % Read value function
        Q = par.Q;
        
        x = swingup_initial_state();
        
        ta = zeros(length(0:par.simstep:par.simtime), 1);
        xa = zeros(numel(ta), numel(x));
        te = 0;
        
        % Initialize a new trial
        s = discretize_state(x, par);
        a = execute_policy(Q, s, par);

        % Inner loop: simulation steps
        for tt = 1:ceil(par.simtime/par.simstep)
            % Take the chosen action
            TD = max(min(take_action(a, par), par.maxtorque), -par.maxtorque);

            % Simulate a time step
            [te,x] = body_straight([te te+par.simstep],x,TD,par);

            % Save trace
            ta(tt) = te;
            xa(tt, :) = x;
            %te         % takes care of task 2.7a

            s = discretize_state(x, par);
            a = execute_policy(Q, s, par);

            % Stop trial if state is terminal
            if is_terminal(s, par)
                break
            end
        end

        ta = ta(1:tt);
        xa = xa(1:tt, :);
        
    elseif strcmp(par.run_type, 'verify')
        %%
        % Get pointers to functions
        learner.get_parameters = @get_parameters;
        learner.init_Q = @init_Q;
        learner.discretize_state = @discretize_state;
        learner.execute_policy = @execute_policy;
        learner.observe_reward = @observe_reward;
        learner.is_terminal = @is_terminal;
        learner.update_Q = @update_Q;
        learner.take_action = @take_action;
        par.learner = learner;
    end
    
end

% ******************************************************************
% *** Edit below this line                                       ***
% ******************************************************************
function par = get_parameters(par)
    % TOREVIEW: set the values
    par.epsilon = 0.1;        % Random action rate
    %par.gamma = 0.99;       % Discount rate
    par.gamma = 0.99;       % for Task 2.7d
    par.alpha = 0.25;          % Learning rate
    par.pos_states = 31;     % Position discretization
    par.vel_states = 31;     % Velocity discretization
    par.actions = 5;        % Action discretization
    par.trials = 2000;         % Learning trials
end

function Q = init_Q(par)
    % TOREVIEW: Initialize the Q table.
    Q = 5 * ones(par.pos_states, par.vel_states, par.actions);
end

function s = discretize_state(x, par)
    % TOREVIEW: Discretize state. Note: s(1) should be (-1 and +1 necessary for output since first state = last state)
    % scaling
    s(1) = (par.pos_states-1)/2/pi * mod(x(1),2*pi) + 1;   % pi rad must equal states/2 & 0rad must equal 1st state     
    s(1) = round(s(1));     % round for classifying in 1 state
    % TOREVIEW: position, s(2) velocity. Assuming that x(2) will give us the
    % velocity 
    x(2) = max(min(x(2),5*pi),-5*pi) + 5*pi;     % clip inputs and scale [0,10pi]
    s(2) = (par.pos_states-1)/10/pi * x(2) + 1;      % scaling (+1 is for going from 0 -> 1 as first number)
    s(2) = round(s(2));     % round for classifying in 1 state
end

function u = take_action(a, par)
    % TOREVIEW: Calculate the proper torque for action a. This cannot
    % TOREVIEW: exceed par.maxtorque.
    u = (a-1)*2*par.maxtorque/(par.actions-1) - par.maxtorque;   % calculate positive and then shift, # actions - 1 steps in between   
end

function r = observe_reward(a, sP, par)
    % TOREVIEW: Calculate the reward for taking action a,
    % TOREVIEW: resulting in state sP.
    % take previous state if reward is never reached due to terminal state
    if sP == 16
        r = 1;
    else
        r = 0;
    end
end

function t = is_terminal(sP, par)
    % TOREVIEW: Return 1 if state sP is terminal, 0 otherwise.
    % same as reward, since we don't want to learn how to balance, we want
    % to learn how to swing upwards
    if sP == 16
        t = 1;
    else
        t = 0;
    end
end


function a = execute_policy(Q, s, par)
    % TOREVIEW: Select an action for state s using the
    % TOREVIEW: epsilon-greedy algorithm.
    p = rand(1);    % some random prob
    if p < par.epsilon
        a = randi([1 par.actions]);
    else
        % could give problems once there are multiple entries with max
        % value
        Q = Q(s(1),s(2),:);         % only look at optimum action for current state
        M = max(Q,[],[1 2]);        % calculates maximum Q value per action channel
        [~,a] = max(M);              % first arg = value and 2nd arg = index
        
    end 
    % For task 2.7c
    %Q = Q(s(1),s(2),:);         % only look at optimum action for current state
    %M = max(Q,[],[1 2]);        % calculates maximum Q value per action channel
    %[~,a] = max(M);              % first arg = value and 2nd arg = index
end

function Q = update_Q(Q, s, a, r, sP, aP, par)
    % TOREVIEW: Implement the SARSA update rule.
    Q(s(1),s(2),a) = Q(s(1),s(2),a) + par.alpha*(r + par.gamma*Q(sP(1),sP(2),aP) - Q(s(1),s(2),a));
end

