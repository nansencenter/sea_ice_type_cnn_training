from utility import FileBasedConfigure, MemoryBasedConfigure,  read_input_params


def main():

    archive_ = read_input_params()
    if archive_.memory_mode:
        config_ = MemoryBasedConfigure(archive=archive_)
    else:
        config_ = FileBasedConfigure(archive=archive_)

    if archive_.apply_instead_of_training:
        config_.apply_model()
    else:
        config_.train_model()


if __name__ == "__main__":
    main()
