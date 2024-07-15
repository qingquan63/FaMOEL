import numpy as np
import geatpy as ea


def update_CA(CA, MaxSize):
    """
    function CA = UpdateCA(CA,New,MaxSize)
        CA = [CA,New];
        N  = length(CA);
        if N <= MaxSize
            return;
        end

        %% Calculate the fitness of each solution
        CAObj = CA.objs;
        CAObj = (CAObj-repmat(min(CAObj),N,1))./(repmat(max(CAObj)-min(CAObj),N,1));
        I = zeros(N);
        for i = 1 : N
            for j = 1 : N
                I(i,j) = max(CAObj(i,:)-CAObj(j,:));
            end
        end
        C = max(abs(I));
        F = sum(-exp(-I./repmat(C,N,1)/0.05)) + 1;

        %% Delete part of the solutions by their fitnesses
        Choose = 1 : N;
        while length(Choose) > MaxSize
            [~,x] = min(F(Choose));
            F = F + exp(-I(Choose(x),:)/C(Choose(x))/0.05);
            Choose(x) = [];
        end
        CA = CA(Choose);
    end
    """
    # CA = np.vstack((CA, New))
    N = CA.shape[0]
    if N <= MaxSize:
        Choose = np.arange(N)
        return Choose[0]

    # Calculate the fitness of each solution
    CA = (CA - np.min(CA, axis=0)) / (np.max(CA, axis=0) - np.min(CA, axis=0))
    I = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            I[i, j] = np.max((CA[i, :] - CA[j, :]))

    C = np.max(np.abs(I), axis=0)
    F = np.sum(-np.exp(-I/C/0.05), axis=0) + 1

    # Delete part of the solutions by their fitness
    Choose = list(np.arange(N))
    while len(Choose) > MaxSize:
        x = np.argmin(F[Choose])
        F = F + np.exp(-I[Choose[x]]/C[Choose[x]]/0.05)
        Choose.pop(x)

    return Choose


def update_DA(DA, MaxSize, p):
        """
        DA = [DA;New];
        ND = NDSort(DA,1);
        DA = DA(ND==1, :);
        N  = size(DA, 1);
        if N <= MaxSize
            return;
        end

        %% Select the extreme solutions first
        Choose = false(1,N);
        [~,Extreme1] = min(DA,[],1);
        [~,Extreme2] = max(DA,[],1);
        Choose(Extreme1) = true;
        Choose(Extreme2) = true;

        %% Delete or add solutions to make a total of K solutions be chosen by truncation
        if sum(Choose) > MaxSize
            % Randomly delete several solutions
            Choosed = find(Choose);
            k = randperm(sum(Choose),sum(Choose)-MaxSize);
            Choose(Choosed(k)) = false;
        elseif sum(Choose) < MaxSize
            % Add several solutions by truncation strategy
            Distance = inf(N);
            for i = 1 : N-1
                for j = i+1 : N
                    Distance(i,j) = norm(DA(i).obj-DA(j).obj,p);
                    Distance(j,i) = Distance(i,j);
                end
            end
            while sum(Choose) < MaxSize
                Remain = find(~Choose);
                [~,x]  = max(min(Distance(~Choose,Choose),[],2));
                Choose(Remain(x)) = true;
            end
        end
        Choose = find(Choose);
        """
        # DA = np.vstack((DA, New))
        DA = (DA - np.min(DA, axis=0)) / (np.max(DA, axis=0) - np.min(DA, axis=0))
        Total = np.arange(DA.shape[0])
        ND = ea.ndsortTNS(DA)
        # print(np.sum(ND[0] == 1))
        if np.sum(ND[0] == 1) <= MaxSize:
            levels = ND[0]
            dis = ea.crowdis(DA, levels)  # 计算拥挤距离
            FitnV = np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort')  # 计算适应度
            chooseFlag = ea.selecting('dup', FitnV.reshape(-1, 1), MaxSize)
            return chooseFlag
        DA = DA[ND[0] == 1, :]
        Total = Total[ND[0] == 1]
        N = DA.shape[0]

        # Select the extreme solutions first
        Choose = np.arange(N) < -1
        max_idx = np.argmax(DA, axis=0)
        min_idx = np.argmin(DA, axis=0)

        Choose[max_idx] = True
        Choose[min_idx] = True

        if np.sum(Choose) > MaxSize:
            Choosed = np.where(Choose)
            k = np.random.randint(np.sum(Choose), size=np.sum(Choose)-MaxSize)
            k = k.astype(int)
            Choose[Choosed[0][k]] = False
        elif np.sum(Choose) < MaxSize:
            Distance = np.zeros((N, N))
            Distance[Distance == 0] = np.inf
            for i in range(N-1):
                for j in range(i+1, N):
                    Distance[i, j] = np.linalg.norm(DA[i, :]-DA[j, :], ord=p)
                    Distance[j, i] = Distance[i, j]

            while np.sum(Choose) < MaxSize:
                Remain = np.where(~Choose)
                temp = Distance[Remain[0], :]
                Remain1 = np.where(Choose)
                temp = temp[:, Remain1[0]]
                x = np.argmax(np.min(temp, axis=1))
                Choose[Remain[0][x]] = True

        Choose = np.where(Choose)

        Total = Total[Choose]
        return Total


if __name__ == '__main__':
    # obj = np.loadtxt('F:\Download\ObjectiveReduction-master\obj.txt')
    CA = np.loadtxt('F:\Download\ObjectiveReduction-master\\CA.txt')
    new = np.loadtxt('F:\Download\ObjectiveReduction-master\\new.txt')
    # CA = np.random.rand(100, 3)
    # new = np.random.rand(100, 3)
    # res = update_CA(CA, new, 5)
    res = update_DA(CA, new, 50, 1.0/10)
    print(res)
    print('over')