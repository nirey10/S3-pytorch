import torch
import S3utils
import torch.nn.functional as F


def compute_style_loss(style_features, target_features):
    # the style loss
    # initialize the style loss to 0
    style_loss = 0
    style_grams = {layer: batch_gram_matrix(style_features[layer]) for layer in style_features}
    # then add to it for each layer's gram matrix loss
    for layer in S3utils.style_weights:
        # get the "target" style representation for the layer
        target_feature = target_features[layer]
        target_gram = batch_gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        # get the "style" style representation
        style_gram = style_grams[layer]
        # the style loss for one layer, weighted appropriately
        layer_style_loss = S3utils.style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
        # add to the style loss
        style_loss += layer_style_loss / (d * h * w)

    return style_loss

def compute_style_loss_2(style_features, target_features):
    # the style loss
    # initialize the style loss to 0
    style_loss = 0
    style_grams = {layer: batch_gram_matrix(style_features[layer]) for layer in style_features}
    # then add to it for each layer's gram matrix loss
    for layer in S3utils.style_weights_2:
        # get the "target" style representation for the layer
        target_feature = target_features[layer]
        target_gram = batch_gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        # get the "style" style representation
        style_gram = style_grams[layer]
        # the style loss for one layer, weighted appropriately
        layer_style_loss = S3utils.style_weights_2[layer] * torch.mean((target_gram - style_gram) ** 2)
        # add to the style loss
        style_loss += layer_style_loss / (d * h * w)

    return style_loss

def compute_style_loss_same_weight(style_features, target_features):
    # the style loss
    # initialize the style loss to 0
    style_loss = 0
    style_grams = {layer: batch_gram_matrix(style_features[layer]) for layer in style_features}
    # then add to it for each layer's gram matrix loss
    for layer in S3utils.style_weights:
        # get the "target" style representation for the layer
        target_feature = target_features[layer]
        target_gram = batch_gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        # get the "style" style representation
        style_gram = style_grams[layer]
        # the style loss for one layer, weighted appropriately
        layer_style_loss = torch.mean((target_gram - style_gram) ** 2)
        # add to the style loss
        style_loss += layer_style_loss / (d * h * w)

    return style_loss

def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """

    # get the batch_size, depth, height, and width of the Tensor
    _, d, h, w = tensor.size()

    # reshape so we're multiplying the features for each channel
    tensor = tensor.view(d, h * w)

    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())

    return gram

def batch_gram_matrix(img):
    """
    Compute the gram matrix by converting to 2D tensor and doing dot product
    img: (batch, channel/depth, height, width)
    """
    b, d, h, w = img.size()
    img = img.view(b*d, h*w) # fix the dimension. It doesn't make sense to put b=1 when it's not always the case
    gram = torch.mm(img, img.t())
    return gram

def compute_content_loss(target_feature, content_feature):
    return torch.mean((target_feature - content_feature)**2)

def compute_content_loss2(target_feature, content_feature):
    return F.mse_loss(target_feature, content_feature)

def compute_content_loss_individual(target_feature, content_feature):
    total_loss = 0
    for target_feature_single, content_feature_single in zip(target_feature, content_feature):
        total_loss += torch.mean((target_feature_single - content_feature_single)**2)
    return total_loss

def total_variation_loss(img, weight=5e-2):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return weight * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)


def tv_loss(img, tv_weight=5e-2):
    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss
