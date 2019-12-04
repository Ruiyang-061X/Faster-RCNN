from torch import nn
from torchvision.models import vgg16


class ExtractorVGG16(nn.Module):

    def __init__(self):
        super(ExtractorVGG16, self).__init__()
        model = vgg16(pretrained=True)
        extractor = list(model.features)[ : 30]
        for layer in extractor[ : 10]:
            for parameter in layer.parameters():
                parameter.requires_grad = False
        self.extractor = nn.Sequential(*extractor)

    def forward(self, x):
        x = self.extractor(x)

        return x

class ClassifierVGG16(nn.Module):

    def __init__(self):
        super(ClassifierVGG16, self).__init__()
        model = vgg16(pretrained=True)
        classifier = list(model.classifier)
        del classifier[6]
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        x = self.classifier(x)

        return x


if __name__ == '__main__':
    model = vgg16(pretrained=True)
    print(list(model.features))
    print(len(list(model.features)))
    print(list(model.features)[0])
    print(list(model.features)[0].parameters())
    for parameter in list(model.features)[0].parameters():
        print(parameter)
    print(list(model.features)[0].weight)
    print(list(model.features)[0].bias)
    print(list(model.features)[0].weight.data)
    print(list(model.features)[0].bias.data)


    for key, value in dict(list(model.features)[0].named_parameters()).items():
        print(key)
        print(value)
        print(value[0].requires_grad)