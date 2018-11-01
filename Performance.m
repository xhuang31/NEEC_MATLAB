function [F1macro,F1micro]=Performance(Xtrain,Xtest,Ytrain,Ytest)
    % Multi class Classification
    if size(Ytrain,2)==1 && length(unique(Ytrain)) > 2
        t = templateSVM('Standardize',true);% 'KernelFunction','linear','rbf','polynomial' , 'KernelScale','auto'
        model=fitcecoc(Xtrain,Ytrain,'Learners',t);
        pred_label=predict(model,Xtest);
        % model = svmtrain(Ytrain,Xtrain,'-q');
        % [predict_label, ~, ~] = svmpredict(Ytest, Xtest, model,'-q');
        [micro, macro] = micro_macro_PR(pred_label,Ytest);

        F1macro = macro.fscore;
        F1micro = micro.fscore;
       
    else
        rng default % For repeatability
        % For multi-label classification, computer micro and macro
        NumLabel=size(Ytest,2);
        macroTP=zeros(NumLabel,1);
        macroFP=zeros(NumLabel,1);
        macroFN=zeros(NumLabel,1);
        macroF=zeros(NumLabel,1);
        for i=1:NumLabel
            model=fitcsvm(Xtrain,Ytrain(:,i),'Standardize',true,'KernelFunction','RBF','KernelScale','auto'); %
            pred_label=predict(model,Xtest);
            mat=confusionmat(Ytest(:,i), pred_label);
            if size(mat,1)==1
                macroTP(i)=sum(pred_label);
                macroFP(i)=0;
                macroFN(i)=0;
                if macroTP(i)~=0
                    macroF(i)=1;
                end
            else
                macroTP(i)=mat(2,2);
                macroFP(i)=mat(1,2);
                macroFN(i)=mat(2,1);
                macroF(i)=2*macroTP(i)/(2*macroTP(i)+macroFP(i)+macroFN(i));
            end  
        end
        F1macro = mean(macroF);
        F1micro = 2*sum(macroTP)/(2*sum(macroTP)+sum(macroFP)+sum(macroFN));
    end
end